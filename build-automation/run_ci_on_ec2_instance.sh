#!/bin/bash

#This script starts build processes on EC2 GPU instance.


if [ "$#" -ne 3 ]; then
    echo "$0 requires 3 arguments."
    echo "Usage:"
    echo "$0 [Tar archive of repo] [EC2 instance id] [AWS CLI profile]"
    exit 1
fi

repo_archive="$1"
ec2_instance_id="$2"
aws_cli_profile="$3"

if [[ -z "$repo_archive" ]]; then
    echo "No repo archive supplied."
    exit 15
fi

function is_error_then_quit {
	ret_val=$?
	if [ $ret_val -ne 0 ]; then
		echo "Exiting because of error no $ret_val"
		exit $ret_val
	fi
}

function is_error_then_quit_and_shutdown {
	ret_val=$?
	if [ $ret_val -ne 0 ]; then
		echo "Exiting because of error no $ret_val"
                aws ec2 stop-instances --instance-ids $ec2_instance_id --profile $aws_cli_profile
		exit $ret_val
	fi
}

set -x

aws ec2 start-instances --instance-ids $ec2_instance_id --profile $aws_cli_profile
is_error_then_quit
sleep 60
ip_address=$(aws ec2 describe-instances --instance-ids $ec2_instance_id --profile $aws_cli_profile | grep PublicDnsName | head -1 | awk -F'[:]' '{print $2}' | xargs)
is_error_then_quit_and_shutdown
ip_address=${ip_address%?}
is_error_then_quit_and_shutdown
echo $ip_address

scp -o ConnectTimeout=120 -oStrictHostKeyChecking=no build-automation/build_and_run_in_container.sh ubuntu@$ip_address:/home/ubuntu/
is_error_then_quit_and_shutdown
scp -o ConnectTimeout=120 -oStrictHostKeyChecking=no build-automation/build_and_run_ci.sh ubuntu@$ip_address:/home/ubuntu/
is_error_then_quit_and_shutdown
scp -oStrictHostKeyChecking=no $repo_archive ubuntu@$ip_address:/home/ubuntu/
is_error_then_quit_and_shutdown
ssh -oStrictHostKeyChecking=no ubuntu@$ip_address /home/ubuntu/build_and_run_in_container.sh
run_result=$?
scp -r -oStrictHostKeyChecking=no ubuntu@$ip_address:"/home/ubuntu/unit-test-results /home/ubuntu/integration-test-results"  build/test-reports/ 
test_results_copy_result=$?
scp -r -oStrictHostKeyChecking=no ubuntu@$ip_address:"/home/ubuntu/ci-artifacts"  build/ci-artifacts
ls -la build/ci-artifacts
aws ec2 stop-instances --instance-ids $ec2_instance_id --profile $aws_cli_profile
exit $run_result

