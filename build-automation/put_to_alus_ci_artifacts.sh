#!/bin/bash

# This script uploads CI artifacts to https://alus-ci-artifacts1.s3.eu-central-1.amazonaws.com/

if [ "$#" -ne 3 ]; then
    echo "$0 requires 3 arguments."
    echo "Usage:"
    echo "$0 [build name] [artifact] [AWS CLI profile]"
    exit 1
fi

build_name="$1"
artifact="$2"
aws_cli_profile="$3"
bucket_name="alus-ci-artifacts1"
ci_folder_prefix="bitbucket_ci"
object_key="$ci_folder_prefix/$build_name/$(basename -- $artifact)"

function is_error_then_quit {
	ret_val=$?
	if [ $ret_val -ne 0 ]; then
		echo "Exiting because of error no $ret_val"
		exit $ret_val
	fi
}

aws s3api put-object --bucket $bucket_name --key $object_key  --body $artifact --acl public-read --storage-class STANDARD_IA --profile $aws_cli_profile
is_error_then_quit
echo "Uploaded resource available at https://alus-ci-artifacts1.s3.eu-central-1.amazonaws.com/$object_key"
