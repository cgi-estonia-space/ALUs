#!/bin/bash

set -e

if [ "$#" -lt 2 ]; then
    echo "$0 "
    echo "Usage:"
    echo "$0 [ALUs source or package] [artifacts and test results output location] <resources folder - optional>"
fi

function is_error_then_quit {
	ret_val=$?
	if [ $ret_val -ne 0 ]; then
		echo "Exiting because of error no $ret_val"
		exit $ret_val
	fi
}

alus_package=$1
ci_output_loc=$2
container_work_dir="/root/alus"
alus_package_filename_w_ext=${alus_package##*/}
alus_package_filename=${alus_package_filename_w_ext%%.*}
build_id=$(echo $alus_package_filename | rev | cut -d"_" -f1 | rev)

docker pull cgialus/alus-devel:latest

container_name="alus_$build_id"
set +e
docker stop $container_name 2> /dev/null
docker rm $container_name 2> /dev/null
set -e

docker run -t -d --gpus all --name $container_name cgialus/alus-devel
docker exec -t $container_name mkdir $container_work_dir
docker cp $alus_package $container_name:$container_work_dir/
docker exec -t $container_name bash -c "tar -xzf $container_work_dir/$alus_package_filename_w_ext -C $container_work_dir/"
local_resource_folder=$3
if [[ $local_resource_folder ]]; then
  docker cp $local_resource_folder $container_name:$container_work_dir/
fi

results_dir=$ci_output_loc/$alus_package_filename
mkdir -p $results_dir

set +e
docker exec -t $container_name bash -c "cd $container_work_dir; CUDAARCHS=70 build-automation/build_and_run_ci.sh"
tests_return_value1=$?
set -e
docker cp $container_name:$container_work_dir/build/unit-test/test-results/. $results_dir
docker cp $container_name:$container_work_dir/build/test-integration/test-results/. $results_dir

set +e
docker exec -t $container_name bash -c "cd $container_work_dir; CC=clang CXX=clang++ CUDAARCHS=70 build-automation/build_and_run_ci.sh"
tests_return_value2=$?
set -e
mkdir -p $results_dir/clang
docker cp $container_name:$container_work_dir/build/unit-test/test-results/. $results_dir/clang
docker cp $container_name:$container_work_dir/build/test-integration/test-results/. $results_dir/clang

# Stash resources to local machine so no need to redownload (some of) those next time
if [[ $local_resource_folder ]]; then
  docker cp $container_name:$container_work_dir/resources/. $local_resource_folder/
fi

docker stop $container_name
docker rm $container_name
exit $(($tests_return_value1 | $tests_return_value2))
