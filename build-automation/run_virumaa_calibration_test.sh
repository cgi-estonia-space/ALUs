#!/usr/bin/env bash

set -e
set -x

function print_help() {
  echo "Usage: "
  echo "$0 <test data folder>"
}

if [[ "$#" != 1 ]]; then
  echo "Wrong count of input arguments"
  print_help
  exit 1
fi

test_data_folder=$1

time ./alus --alg_name sentinel1-calibrate -i "$test_data_folder"/S1A_IW_SLC__1SDV_20180815T154813_20180815T154840_023259_028747_4563_thin.SAFE -o /tmp/ -x 5000 -y 5000 --aux "$test_data_folder"/S1A_IW_SLC__1SDV_20180815T154813_20180815T154840_023259_028747_4563_thin.SAFE -p "subswath=IW1,polarisation=VV,calibration_type=beta"
