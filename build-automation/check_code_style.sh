#!/usr/bin/env bash

# Checks code style in alus repository

repository_path=$1
destination_branch=$2

if [ "$repository_path" = "" ]; then
  echo "No repository is provided. Please enter a repository path as a first argument."
  exit 2
fi

if [ ! -d "$repository_path" ]; then
  printf "No such directory %s\n" "$repository_path"
  exit 2
fi

if [ "$destination_branch" = "" ]; then
  echo "No destination branch is provided. Please enter destination branch as the second argument,"
  exit 2
fi

git show-ref --verify --quiet refs/heads/"$destination_branch"
tt=$?
if [[ ! "$tt" ]]; then
  echo "No such branch found. Please enter the correct branch.."
  exit 2
fi

# Check file extensions
./build-automation/check_file_extensions.sh "$repository_path" "$destination_branch"
extensions_result=$?

# Check file names
echo ""
./build-automation/check_file_names.sh "$repository_path" "$destination_branch"
names_result=$?

# Check cmake target names
echo ""
./build-automation/check_cmake_targets.sh "$repository_path" "$destination_branch"
cmake_result=$?

# Check for correct license
echo ""
./build-automation/check_license_texts.sh "$repository_path" "$destination_branch"
license_text_result=$?

# Check for correct formatting
echo ""
./build-automation/check_clang_format.sh "$repository_path" "$destination_branch"
clang_format_result=$?

# Check for code style using clang-style
echo ""
./build-automation/check_clang_style.sh "$repository_path" "$destination_branch" build
clang_style_result=$?

exit $((extensions_result | names_result | cmake_result | license_text_result | clang_format_result | clang_style_result))
