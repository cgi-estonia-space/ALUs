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
# cuDNN libraries cannot be distributed openly due to ethical AI concerns. So host machine that runs this must install it on the docker instance in order to Tensorflow be able to run.
docker cp /home/ubuntu/libcudnn7_7.6.5.32-1+cuda10.2_amd64.deb alus_container:/tmp/
is_error_then_quit
docker exec -t alus_container bash -c "apt install -y /tmp/libcudnn7_7.6.5.32-1+cuda10.2_amd64.deb"
is_error_then_quit
docker exec -t alus_container bash -c "tar -xzf /root/alus/*.tar.gz -C /root/alus/"
is_error_then_quit
docker exec -t alus_container bash -c "cd /root/alus; build-automation/build_and_run_ci.sh"
tests_return_value=$?
rm -rf ~/unit-test-results
rm -rf ~/integration-test-results
rm -rf ~/ci-artifacts
mkdir ~/ci-artifacts
docker cp alus_container:/root/alus/build/unit-test/test-results ~/unit-test-results
is_error_then_quit
docker cp alus_container:/root/alus/build/test_integration/test-results ~/integration-test-results
is_error_then_quit
docker exec -t alus_container bash -c "gdal_translate -of PNG -ot Byte -scale /tmp/tc_test.tif /tmp/tc_test.png"
docker cp alus_container:/tmp/tc_test.png ~/ci-artifacts/.
docker exec -t alus_container bash -c "gdal_translate -of PNG -ot Byte -scale /tmp/4_bands_coh.tif /tmp/4_bands_coh.png"
docker cp alus_container:/tmp/4_bands_coh.png ~/ci-artifacts/.
docker stop alus_container
docker rm alus_container
exit $tests_return_value

