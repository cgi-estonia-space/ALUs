#!/bin/bash

set -e

function print_help {
    echo "Usage:"
    echo "$0 <test data folder> <dem files location> [optional - output folder]"
}

if [ $# -lt 2 ]; then
    echo "Wrong count of input arguments"
    print_help
    exit 1
fi

test_dataset_dir=$1
dem_files_dir=$2

output_dir=$3
if [ $# -eq 2 ]; then
  me=$0
  # foo.ext
  me=${me##*/}
  # foo
  me=${me%.*}
  output_dir="/tmp/$me"
  echo "Created output folder for results - $output_dir"
fi

mkdir -p $output_dir

test_1_prod_path=$output_dir/S1A_IW_SLC__1SDV_20180815T154813_20180815T154840_023259_028747_4563_Calib_IW1_tc.tif
time alus --alg_name calibration-routine -i $test_dataset_dir/S1A_IW_SLC__1SDV_20180815T154813_20180815T154840_023259_028747_4563.SAFE \
     -o $test_1_prod_path \
     -x 4000 -y 4000 -p "subswath=IW1,polarisation=VV,calibration_type=beta" --dem $dem_files_dir/srtm_42_01.tif

if [[ -z "${NIGHTLY_GOLDEN_DIR}" ]]; then
  echo "no golden directory defined, no verification executed"
  exit 0
fi

echo "Validating $test_1_prod_path"
./alus_result_check.py -I $test_1_prod_path -G "$NIGHTLY_GOLDEN_DIR"/S1A_IW_SLC__1SDV_20180815T154813_20180815T154840_023259_028747_4563_Calib_IW1_tc.tif

exit $?