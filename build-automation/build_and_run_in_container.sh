#!/bin/bash

# This script runs on EC2 GPU instance.

set -x

function is_error_then_quit {
	ret_val=$?
	if [ $ret_val -ne 0 ]; then
		echo "Exiting because of error no $ret_val"
		sudo shutdown -h +1 &
		exit $ret_val
	fi
}

docker stop "$(docker ps -a -q)"
docker rm "$(docker ps -a -q)"

docker pull cgialus/alus-infra:latest
is_error_then_quit
docker run -t -d --gpus all --name alus_container cgialus/alus-infra
is_error_then_quit
docker exec -t alus_container mkdir /root/alus
is_error_then_quit
docker cp ~/*.tar.gz alus_container:/root/alus/
is_error_then_quit
docker cp ~/resources alus_container:root/alus/
docker exec -t alus_container bash -c "tar -xzf /root/alus/*.tar.gz -C /root/alus/"
is_error_then_quit
docker exec -t alus_container bash -c "cd /root/alus; build-automation/build_and_run_ci.sh"
tests_return_value=$?
# Stash resources to local machine so no need to redownload (some of) those next time
docker cp alus_container:/root/alus/resources ~/
rm -rf ~/unit-test-results
rm -rf ~/integration-test-results
rm -rf ~/ci-artifacts
mkdir ~/ci-artifacts
docker cp alus_container:/root/alus/build/unit-test/test-results ~/unit-test-results
is_error_then_quit
docker cp alus_container:/root/alus/build/test_integration/test-results ~/integration-test-results
is_error_then_quit
docker exec -t alus_container bash -c "gdal_translate -of PNG -ot Byte -scale /tmp/4_bands_cuda_coh.tif /tmp/4_bands_cuda_coh.png"
docker cp alus_container:/tmp/4_bands_cuda_coh.png ~/ci-artifacts/.
docker exec -t alus_container bash -c "gdal_translate -of PNG -ot Byte -scale 0 1.0 0 255 /tmp/tc_beirut_test.tif /tmp/tc_beirut_test.png"
docker cp alus_container:/tmp/tc_beirut_test.png ~/ci-artifacts/.
docker stop alus_container
docker rm alus_container
exit $tests_return_value

