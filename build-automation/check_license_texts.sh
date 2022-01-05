#!/usr/bin/env bash

license_text=" * This program is free software; you can redistribute it and/or modify it
               * under the terms of the GNU General Public License as published by the Free
               * Software Foundation; either version 3 of the License, or (at your option)
               * any later version.
               * This program is distributed in the hope that it will be useful, but WITHOUT
               * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
               * FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for
               * more details.
               * You should have received a copy of the GNU General Public License along
               * with this program; if not, see http://www.gnu.org/licenses/"

repository_path=$1
destination_branch=$2

exclude_regex="$repository_path.*\/(((external|build|cmake-build-.*|\.git|\.idea|resources|docs|goods)\/.*)|(.*\.(txt|sh|py|md)))"
include_regex='.*\/(app|sentinel1|snap-engine|test|test-integration|util)/.*\.(cc|h|cuh|cu)'

error_count=0

function should_file_be_ignored() {
  # This list should be used when non-standard license is being used, e.g. Nvidia license.
  ignore_list=("helper_cuda.h")

  file=$1
  filename="${file##*/}"
  if echo "${ignore_list[*]}" | grep -w -q "$filename"; then
    return 0
  fi

  return 1
}

if [ "$repository_path" = "" ]; then
  echo "No repository is provided. Please enter a repository path as a first argument."
  exit 1
fi

if [ ! -d "$repository_path" ]; then
  printf "No such directory %s\n" "$repository_path"
  exit 1
fi

printf "Checking for license texts in directory %s...\n" "$repository_path"

# Get list of changed files
base=$(git merge-base refs/remotes/origin/"$destination_branch" HEAD)
modified_filepaths=()

while IFS='' read -r line; do
  absolute_filepath=$(realpath "$line")

  modified_filepaths+=("$absolute_filepath")
done < <(git diff-tree --no-commit-id --diff-filter=d --name-only -r "$base" HEAD)

while IFS= read -r -d '' file; do
  if ! grep -q "$license_text" "$file"; then
    if should_file_be_ignored "$file"; then
      continue
    fi
    ((error_count++))
    if ((error_count == 1)); then
      printf >&2 "\nFollowing files do not have license text:\n"
    fi
    echo >&2 "$file"
  fi
done < <(find "${modified_filepaths[@]}" -regextype posix-extended -type f -regex "$exclude_regex" -prune -o -regex "$include_regex" -type f -print0)

if ((error_count > 0)); then
  exit 1
fi

echo "All files include license texts."
exit 0
