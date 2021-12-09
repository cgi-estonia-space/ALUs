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

test_1_prod_path=$output_dir/S1A_IW_SLC__1SDV_20210722T005537_20210722T005604_038883_049695_2E58_Calib_b26_tc.tif
time alus --alg_name calibration-routine -i $test_dataset_dir/S1A_IW_SLC__1SDV_20210722T005537_20210722T005604_038883_049695_2E58.SAFE \
     -o $test_1_prod_path \
     -x 2000 -y 2000 -p "subswath=IW2,polarisation=VV,calibration_type=gamma,first_burst_index=2,last_burst_index=6" \
     --dem $dem_files_dir/srtm_51_09.tif --dem $dem_files_dir/srtm_52_09.tif

test_2_prod_path=$output_dir/S1A_IW_SLC__1SDV_20210722T005537_20210722T005604_038883_049695_2E58_Calib_b26_tc_zipped.tif
time alus --alg_name calibration-routine -i $test_dataset_dir/S1A_IW_SLC__1SDV_20210722T005537_20210722T005604_038883_049695_2E58.zip \
     -o $test_2_prod_path \
     -x 2000 -y 2000 -p "subswath=IW2,polarisation=VV,calibration_type=gamma,first_burst_index=2,last_burst_index=6" \
     --dem $dem_files_dir/srtm_51_09.zip --dem $dem_files_dir/srtm_52_09.zip

if [[ -z "${NIGHTLY_GOLDEN_DIR}" ]]; then
  echo "no golden directory defined, no verification executed"
  exit 0
fi

echo "Validating $test_1_prod_path"
./alus_result_check.py -I $test_1_prod_path -G "$NIGHTLY_GOLDEN_DIR"/S1A_IW_SLC__1SDV_20210722T005537_20210722T005604_038883_049695_2E58_Calib_b26_tc.tif
echo "Validating $test_2_prod_path"
./alus_result_check.py -I $test_2_prod_path -G "$NIGHTLY_GOLDEN_DIR"/S1A_IW_SLC__1SDV_20210722T005537_20210722T005604_038883_049695_2E58_Calib_b26_tc.tif

exit $?