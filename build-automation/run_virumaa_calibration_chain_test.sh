#!/bin/bash

set -e
set -x

function print_help {
    echo "Usage:"
    echo "$0 <test data folder> <dem files location>"
}

if [[ "$#" != 2 ]]; then
    echo "Wrong count of input arguments"
    print_help
    exit 1
fi

test_data_folder=$1
dem_data_folder="$2/SRTM3/"


time ./alus --alg_name calibration-routine -i $test_data_folder/S1A_IW_SLC__1SDV_20180815T154813_20180815T154840_023259_028747_4563_thin.SAFE -o /tmp/alus_S1A_IW_SLC__1SDV_20180815T154813_20180815T154840_023259_028747_4563_Calib_tc.tif -x 5000 -y 5000 -p "subswath=IW1,polarisation=VV,calibration_type=beta" --dem $dem_data_folder/srtm_42_01.tif
gdalinfo -checksum /tmp/alus_S1A_IW_SLC__1SDV_20180815T154813_20180815T154840_023259_028747_4563_Calib_tc.tif
./alus_result_check.py -I /tmp/alus_S1A_IW_SLC__1SDV_20180815T154813_20180815T154840_023259_028747_4563_Calib_tc.tif -G "$NIGHTLY_GOLDEN_DIR"/S1A_IW_SLC__1SDV_20180815T154813_20180815T154840_023259_028747_4563_Calib_tc.tif
