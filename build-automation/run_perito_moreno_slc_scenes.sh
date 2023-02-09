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

test_1_prod_path=$output_dir/S1A_IW_SLC__1SDV_20220306T235614_20220306T235642_042207_0507B2_2270_tnr_Cal_IW1_deb_mrg_tc.tif
time alus-cal -i $test_dataset_dir/S1A_IW_SLC__1SDV_20220306T235614_20220306T235642_042207_0507B2_2270.zip \
     -o $test_1_prod_path --ll info \
     -p VV -t sigma \
     --aoi "POLYGON((-72.827919 -51.973301,-69.389046 -51.141209,-70.393578 -49.548401,-73.725533 -50.353531000000004,-72.827919 -51.973301))" \
     --dem "$dem_files_dir/Copernicus_DSM_COG_10_S50_00_W071_00_DEM.tif $dem_files_dir/Copernicus_DSM_COG_10_S50_00_W072_00_DEM.tif" \
     --dem "$dem_files_dir/Copernicus_DSM_COG_10_S51_00_W070_00_DEM.tif $dem_files_dir/Copernicus_DSM_COG_10_S51_00_W071_00_DEM.tif" \
     --dem "$dem_files_dir/Copernicus_DSM_COG_10_S51_00_W072_00_DEM.tif $dem_files_dir/Copernicus_DSM_COG_10_S51_00_W073_00_DEM.tif" \
     --dem "$dem_files_dir/Copernicus_DSM_COG_10_S51_00_W074_00_DEM.tif $dem_files_dir/Copernicus_DSM_COG_10_S52_00_W070_00_DEM.tif" \
     --dem "$dem_files_dir/Copernicus_DSM_COG_10_S52_00_W071_00_DEM.tif $dem_files_dir/Copernicus_DSM_COG_10_S52_00_W072_00_DEM.tif" \
     --dem "$dem_files_dir/Copernicus_DSM_COG_10_S52_00_W073_00_DEM.tif $dem_files_dir/Copernicus_DSM_COG_10_S52_00_W074_00_DEM.tif"

if [[ -z "${NIGHTLY_GOLDEN_DIR}" ]]; then
  echo "no golden directory defined, no verification executed"
  exit 0
fi

set +e

echo "Validating $test_1_prod_path"
./alus_result_check.py -I $test_1_prod_path -G "$NIGHTLY_GOLDEN_DIR"/S1A_IW_SLC__1SDV_20220306T235614_20220306T235642_042207_0507B2_2270_tnr_Cal_IW1_deb_mrg_tc.tif
res1=$?


exit $((res1))
