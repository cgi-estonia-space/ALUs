#!/bin/bash

set -e

if [ "$#" -lt 1 ]; then
    echo "$0 requires atleast EC2 instance ID."
    echo "Usage:"
    echo "$0 [EC2 instance id] <AWS CLI profile - optional> <AWS region - optional, but profile required>"
fi

instance_id="$1"

profile=""
if [[ $2 ]]; then
    profile="--profile $2"
fi

region=""
if [[ $3 ]]; then
    region="--region $3"
fi

aws_cli_format_options="--no-cli-pager --output text"

function shutdown_instance {
    aws ec2 stop-instances --instance-ids $instance_id $profile $region $aws_cli_format_options
}

shutdown_instance