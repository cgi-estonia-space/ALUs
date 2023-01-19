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

test_1_prod_path=$output_dir/S1A_IW_SLC__1SDV_20210722T005537_20210722T005604_038883_049695_2E58_tnr_Calib_IW2_b26_tc.tif
time alus-cal -i $test_dataset_dir/S1A_IW_SLC__1SDV_20210722T005537_20210722T005604_038883_049695_2E58.SAFE \
     -o $test_1_prod_path \
     --sw IW2 --polarisation VV -t gamma --bi1 2 --bi2 6 \
     --dem $dem_files_dir/srtm_51_09.tif --dem $dem_files_dir/srtm_52_09.tif --ll info

test_2_prod_path=$output_dir/S1A_IW_SLC__1SDV_20210722T005537_20210722T005604_038883_049695_2E58_tnr_Cal_IW2_deb_tc_zipped.tif
time alus-cal -i $test_dataset_dir/S1A_IW_SLC__1SDV_20210722T005537_20210722T005604_038883_049695_2E58.zip \
     -o $test_2_prod_path \
     --sw IW2 --polarisation VV -t gamma --bi1 2 --bi2 6 \
     --dem $dem_files_dir/srtm_51_09.zip --dem $dem_files_dir/srtm_52_09.zip --ll info

test_3_prod_path=$output_dir/S1A_IW_SLC__1SDV_20210722T005537_20210722T005604_038883_049695_2E58_tnr_Cal_deb_tc_aoi_merge.tif
time alus-cal -i $test_dataset_dir/S1A_IW_SLC__1SDV_20210722T005537_20210722T005604_038883_049695_2E58.SAFE \
     -o $test_3_prod_path \
     --aoi "POLYGON((16.4519,73.3234 17.2811,73.3234 17.2811,74.6600 16.4519,74.6600))" \
     --polarisation VV -t gamma \
     --dem $dem_files_dir/srtm_51_09.zip --dem $dem_files_dir/srtm_52_09.zip --ll info

test_4_prod_path=$output_dir/S1A_IW_SLC__1SDV_20210722T005537_20210722T005604_038883_049695_2E58_tnr_Cal_deb_tc_aoishp_merge.tif
time alus-cal -i $test_dataset_dir/S1A_IW_SLC__1SDV_20210722T005537_20210722T005604_038883_049695_2E58.SAFE \
     -o $test_4_prod_path \
     --aoi $test_dataset_dir/maharashtra_test4_aoi.shp \
     --polarisation VV -t gamma \
     --dem $dem_files_dir/srtm_51_09.zip --dem $dem_files_dir/srtm_52_09.zip --ll info

if [[ -z "${NIGHTLY_GOLDEN_DIR}" ]]; then
  echo "no golden directory defined, no verification executed"
  exit 0
fi

echo "Validating $test_1_prod_path"
./alus_result_check.py -I $test_1_prod_path -G "$NIGHTLY_GOLDEN_DIR"/S1A_IW_SLC__1SDV_20210722T005537_20210722T005604_038883_049695_2E58_tnr_Calib_b26_tc.tif
echo "Validating $test_2_prod_path"
./alus_result_check.py -I $test_2_prod_path -G "$NIGHTLY_GOLDEN_DIR"/S1A_IW_SLC__1SDV_20210722T005537_20210722T005604_038883_049695_2E58_tnr_Calib_b26_tc.tif
echo "Validating $test_3_prod_path"
./alus_result_check.py -I $test_3_prod_path -G "$NIGHTLY_GOLDEN_DIR"/S1A_IW_SLC__1SDV_20210722T005537_20210722T005604_038883_049695_2E58_tnr_Cal_deb_tc_aoi_merge.tif
echo "Validating $test_4_prod_path"
./alus_result_check.py -I $test_4_prod_path -G "$NIGHTLY_GOLDEN_DIR"/S1A_IW_SLC__1SDV_20210722T005537_20210722T005604_038883_049695_2E58_tnr_Cal_deb_tc_aoi_merge.tif

exit $?
