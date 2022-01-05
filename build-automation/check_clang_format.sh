#!/usr/bin/env bash

repository_path=$1
destination_branch=$2

exclude_regex="$repository_path.*\/(((external|build|cmake-build-.*|\.git|\.idea|resources|docs|goods)\/.*)|(.*\.(txt|sh|py|md|cu|cuh)))"
include_regex='.*\/(app|sentinel1|snap-engine|test|test-integration|util)/.*\.(cc|h|cu|cuh)'

error_count=0

if [ "$repository_path" = "" ]; then
  echo "No repository is provided. Please enter a repository path as a first argument."
  exit 1
fi

if [ ! -d "$repository_path" ]; then
  printf "No such directory %s\n" "$repository_path"
  exit 1
fi

printf "Executing clang-format dry-run for files in directory %s...\n" "$repository_path"

# Get list of changed files
base=$(git merge-base refs/remotes/origin/"$destination_branch" HEAD)
modified_filepaths=()

while IFS='' read -r line; do
  absolute_filepath=$(realpath "$line")

  modified_filepaths+=("$absolute_filepath")
done < <(git diff-tree --no-commit-id --diff-filter=d --name-only -r "$base" HEAD)

while IFS= read -r -d '' file; do
  clang-format --dry-run -Werror "$file"
  if [ $? ]; then
     ((error_count))
  fi
done < <(find "${modified_filepaths[@]}" -regextype posix-extended -type f -regex "$exclude_regex" -prune -o -regex "$include_regex" -type f -print0)

if ((error_count > 0)); then
    exit 1
fi

echo "All files follow the correct formatting."
exit 0
