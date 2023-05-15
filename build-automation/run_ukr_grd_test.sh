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

test_1_prod_path=$output_dir/S1A_IW_GRDH_1SDV_20230130T152052_20230130T152117_047015_05A3B0_874C_tnr_Cal_VV_tc.tif
time alus-cal -i $test_dataset_dir/S1A_IW_GRDH_1SDV_20230130T152052_20230130T152117_047015_05A3B0_874C.SAFE \
     -o $test_1_prod_path --polarisation VV -t gamma --ll info \
     --dem $dem_files_dir/Copernicus_DSM_COG_10_N47_00_E035_00_DEM.tif \
     --dem $dem_files_dir/Copernicus_DSM_COG_10_N47_00_E036_00_DEM.tif \
     --dem $dem_files_dir/Copernicus_DSM_COG_10_N47_00_E037_00_DEM.tif \
     --dem $dem_files_dir/Copernicus_DSM_COG_10_N47_00_E038_00_DEM.tif \
     --dem $dem_files_dir/Copernicus_DSM_COG_10_N47_00_E039_00_DEM.tif \
     --dem $dem_files_dir/Copernicus_DSM_COG_10_N48_00_E035_00_DEM.tif \
     --dem $dem_files_dir/Copernicus_DSM_COG_10_N48_00_E036_00_DEM.tif \
     --dem $dem_files_dir/Copernicus_DSM_COG_10_N48_00_E037_00_DEM.tif \
     --dem $dem_files_dir/Copernicus_DSM_COG_10_N48_00_E038_00_DEM.tif \
     --dem $dem_files_dir/Copernicus_DSM_COG_10_N48_00_E039_00_DEM.tif \
     --dem $dem_files_dir/Copernicus_DSM_COG_10_N49_00_E035_00_DEM.tif \
     --dem $dem_files_dir/Copernicus_DSM_COG_10_N49_00_E036_00_DEM.tif \
     --dem $dem_files_dir/Copernicus_DSM_COG_10_N49_00_E037_00_DEM.tif \
     --dem $dem_files_dir/Copernicus_DSM_COG_10_N49_00_E038_00_DEM.tif \
     --dem $dem_files_dir/Copernicus_DSM_COG_10_N49_00_E039_00_DEM.tif

if [[ -z "${NIGHTLY_GOLDEN_DIR}" ]]; then
  echo "no golden directory defined, no verification executed"
  exit 0
fi

set +e

echo "Validating $test_1_prod_path"
./alus_result_check.py -I $test_1_prod_path -G "$NIGHTLY_GOLDEN_DIR"/S1A_IW_GRDH_1SDV_20230130T152052_20230130T152117_047015_05A3B0_874C_tnr_Cal_VV_tc.tif -O SKIP_ALUs_VERSION

exit $?
