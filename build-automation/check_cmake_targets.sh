#!/usr/bin/env bash

# Checks CMakeLists.txt files for correct target naming

repository_path=$1
destination_branch=$2

exclude_regex="$repository_path.*\/(((external|build|cmake-build-.*|\.git|\.idea|resources|docs|goods)\/.*)|(.*\.(sh|py|md|cuh|cu|cc|h)))"
include_regex='.*\/(app|sentinel1|snap-engine|test|test-integration|util)/.*CMakeLists.txt'

error_count=0

if [ "$repository_path" = "" ]; then
  echo "No repository is provided. Please enter a repository path as a first argument."
  exit 1
fi

if [ ! -d "$repository_path" ]; then
  printf "No such directory %s\n" "$repository_path"
  exit 1
fi


function check_target_names() {
  filename=$1
  case  $(grep -E "add_library|add_executable" "$filename" > /dev/null; echo $?) in
    0)
      command=$(awk '/add_library|add_executable/ {print}' "$filename") # Extracts necessary command from CMakeLists.txt

      # Extract target name
      extraction="${command##*(}"
      read -r -a array <<< "$extraction"
      target_name="${array[0]}"

      # Checks that target name does not contain underscores
      if [[ $target_name =~ .*_.* ]]
      then
        printf "Incorrect target name in file %s -> %s\n" "$filename" "$target_name"
        return 1
      fi

      # No error case
      return 0
      ;;
    1)
      # Nothing to do here
      return 0
      ;;
    2) # This is the error case
      return 2
      ;;
  esac
}


printf "Checking CMake target names in directory %s...\n" "$repository_path"

# Get list of changed files
base=$(git merge-base refs/remotes/origin/"$destination_branch" HEAD)
modified_filepaths=()

while IFS='' read -r line; do
  absolute_filepath=$(realpath "$line")

  modified_filepaths+=("$absolute_filepath")
done < <(git diff-tree --no-commit-id --diff-filter=d --name-only -r "$base" HEAD)

while IFS= read -r -d '' file; do
      check_target_names "$file"
      case $? in
        0)
          ;;
        1)
          ((error_count++))
          ;;
        2)
          echo "Encountered an error during file parsing"
          exit 1
          ;;
      esac
done < <(find "${modified_filepaths[@]}" -regextype posix-extended -type f -regex "$exclude_regex" -prune -o -regex "$include_regex" -type f -print0)

if ((error_count > 0))
then
    exit 1
fi

echo "All CMake targets follow naming guidelines."
exit 0