#!/bin/bash

set -e
set -x

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

# First check if should be started, or already started, then skip to wait for IP address. Or wait until completely stopped
# and then trigger start
status=$(aws ec2 describe-instances --instance-ids $instance_id $profile $region $aws_cli_format_options \
             --query "Reservations[*].Instances[*].{PublicIP:PublicIpAddress,Name:Tags[?Key=='Name']|[0].Value,Status:State.Name}" \
             | cut -f3 -d$'\t')
if [[ $status == "stopped" ]]; then
  echo "Requesting to start instance."
  aws ec2 start-instances --instance-ids $instance_id $profile $region $aws_cli_format_options
elif [[ $status == "shutting-down" || $status == "stopping" ]]; then
  echo "Instance is stopping/shutting-down, waiting until it can be started"
  started=0
  for i in {0..100}
  do
    status=$(aws ec2 describe-instances --instance-ids $instance_id $profile $region $aws_cli_format_options \
      --query "Reservations[*].Instances[*].{PublicIP:PublicIpAddress,Name:Tags[?Key=='Name']|[0].Value,Status:State.Name}" \
      | cut -f3 -d$'\t')
    if [[ $status == "stopped" ]]; then
      echo ""
      echo "Instance is finally stopped, requesting start."
      aws ec2 start-instances --instance-ids $instance_id $profile $region $aws_cli_format_options
      started=1
      break
    fi
    echo -n "."
    sleep 5
  done
  if [[ $started -eq 0 ]]; then
    echo "Timeout occurred for waiting instance to be stopped, try another time."
    exit 1
  fi
elif [[ $status == "" ]]; then
    echo "There is problem with credentials or AWS CLI, please investigate."
    exit 2
fi


started=0
for i in {0..20}
do
  status=$(aws ec2 describe-instances --instance-ids $instance_id $profile $region $aws_cli_format_options \
    --query "Reservations[*].Instances[*].{PublicIP:PublicIpAddress,Name:Tags[?Key=='Name']|[0].Value,Status:State.Name}" \
    | cut -f3 -d$'\t')
  if [[ $status == "running" ]]; then
    echo "Instance is running."
    started=1
    break
  fi
  sleep 5
done

if [[ $started -eq 0 ]]; then
  echo "Timeout reached for starting up EC2 instance"
  shutdown_instance
  exit 1
fi

final_status=$(aws ec2 describe-instances --instance-ids $instance_id $profile $region $aws_cli_format_options \
              --query "Reservations[*].Instances[*].{PublicIP:PublicIpAddress,Name:Tags[?Key=='Name']|[0].Value,Status:State.Name}" \
              | cut -f3 -d$'\t')
echo "$final_status"
