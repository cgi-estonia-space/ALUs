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

function get_instance_status {
  echo $(aws ec2 describe-instances --instance-ids $instance_id $profile $region $aws_cli_format_options \
       --query "Reservations[*].Instances[*].{PublicIP:PublicIpAddress,Name:Tags[?Key=='Name']|[0].Value,Status:State.Name}" \
       | cut -f3 -d$'\t')
}

function start_instance {
  aws ec2 start-instances --instance-ids $instance_id $profile $region $aws_cli_format_options
}

function print_instance_ip {
  echo $(aws ec2 describe-instances --instance-ids $instance_id $profile $region $aws_cli_format_options \
        --query "Reservations[*].Instances[*].{PublicIP:PublicIpAddress,Name:Tags[?Key=='Name']|[0].Value,Status:State.Name}" \
        | cut -f2 -d$'\t')
}

sleep_increment_sec=5

# If is stopped, request start and continue to wait. If shutting-down or stopping, wait, then request start and continue
# to wait. If already running, return IP and exit the script. If no status, there is some more delicate problem.
status=$(get_instance_status)
if [[ $status == "stopped" ]]; then
  echo "Requesting to start instance."
  start_instance
  echo "Instance is booting..."
elif [[ $status == "shutting-down" || $status == "stopping" ]]; then
  echo "Instance is stopping/shutting-down, waiting until it can be started"
  started=0
  timeout_sec=$((5 * 60))
  i_max=$((timeout_sec / sleep_increment_sec))
  for (( i=0; i < i_max; i++ ))
  do
    status=$(get_instance_status)
    if [[ $status == "stopped" ]]; then
      echo ""
      echo "Instance is finally stopped, requesting start."
      start_instance
      echo "Instance is booting..."
      started=1
      break
    fi
    echo -n "."
    sleep $sleep_increment_sec
  done
  if [[ $started -eq 0 ]]; then
    echo "Timeout occurred for waiting instance to be stopped, try another time."
    exit 1
  fi
elif [[ $status == "running" ]]; then
  echo "Instance is already running."
  print_instance_ip
  exit 0
elif [[ $status == "" ]]; then
  echo "There is problem with credentials or AWS CLI, please investigate."
  exit 2
fi

started=0
echo -n "Waiting for instance to be ready (in state 'running')"
timeout_sec=$((2 * 60))
i_max=$((timeout_sec / sleep_increment_sec))
for (( i=0; i < i_max; i++ ))
do
  status=$(get_instance_status)
  if [[ $status == "running" ]]; then
    echo ""
    echo "Instance is running."
    started=1
    break
  fi
  echo -n "."
  sleep $sleep_increment_sec
done


if [[ $started -eq 0 ]]; then
  echo ""
  echo "Timeout reached for starting up EC2 instance"
  shutdown_instance
  exit 3
fi

print_instance_ip
