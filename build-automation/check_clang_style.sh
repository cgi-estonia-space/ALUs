#!/usr/bin/env bash

repository_path=$1
destination_branch=$2
build_dir=$3

exclude_regex="$repository_path.*\/(((external|build|cmake-build-.*|\.git|\.idea|resources|docs|goods)\/.*)|(.*\.(txt|sh|py|md|cu|cuh)))"
include_regex='.*\/(app|sentinel1|snap-engine|test|test-integration|util|algs)/.*\.(cc|h)' # TODO(anton): check that cuda files are analysed as well

file_count=0

set -o pipefail

if [ "$repository_path" = "" ]; then
  echo "No repository is provided. Please enter a repository path as a first argument."
  exit 1
fi

if [ ! -d "$repository_path" ]; then
  printf "No such directory %s\n" "$repository_path"
  exit 1
fi

printf "Executing clang-tidy for files in directory %s...\n" "$repository_path"

# Get list of changed files
base=$(git merge-base refs/remotes/origin/"$destination_branch" HEAD)
modified_filepaths=()

while IFS='' read -r line; do
  absolute_filepath=$(realpath "$line")

  modified_filepaths+=("$absolute_filepath")
done < <(git diff-tree --no-commit-id --diff-filter=d --name-only -r "$base" HEAD)

correct_filepaths=()
while IFS= read -r -d '' file; do
  correct_filepaths+=("$file")
  ((file_count++))
done < <(find "${modified_filepaths[@]}" -regextype posix-extended -type f -regex "$exclude_regex" -prune -o -regex "$include_regex" -type f -print0)
parallel -m -k clang-tidy --format-style=file -extra-arg=-std=c++17 -p "$build_dir" {} ::: "${correct_filepaths[@]}" | tee "$build_dir"/clang-tidy-result.txt
run_result=$?

printf "Scanned %s files." "$file_count"

if [[ ! -d "$build_dir"/test-results ]]; then
  mkdir "$build_dir"/test-results
fi

cat "$build_dir/clang-tidy-result.txt" | ./build-automation/clang-tidy-to-junit.py "$repository_path" >"$build_dir"/test-results/junit-clang-tidy-result.xml

printf "Check results can be found in junit-clang-tidy-result.xml file."
exit $run_result