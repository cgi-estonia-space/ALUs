#!/bin/bash

set -e

function print_help {
    echo "Usage:"
    echo "$0 <test data folder> <COPDEM 30m COG location> <orbit files dir> [optional - output folder]"
}

if [ $# -lt 3 ]; then
    echo "Wrong count of input arguments"
    print_help
    exit 1
fi

test_dataset_dir=$1
dem_files_dir=$2
orbit_dir=$3

output_dir=$4
if [ $# -eq 3 ]; then
  me=$0
  # foo.ext
  me=${me##*/}
  # foo
  me=${me%.*}
  output_dir="/tmp/$me"
  echo "Created output folder for results - $output_dir"
fi

mkdir -p $output_dir

test_1_prod_path=$output_dir/S1A_IW_SLC__1SDV_20210710T005537_20210710T005604_038708_049158_B74A_Orb_Stack_coh_deb_tc.tif
time alus-coh -r $test_dataset_dir/S1A_IW_SLC__1SDV_20210710T005537_20210710T005604_038708_049158_B74A.zip \
     -s $test_dataset_dir/S1A_IW_SLC__1SDV_20210722T005537_20210722T005604_038883_049695_2E58.SAFE \
     -o $test_1_prod_path --sw IW3 -p VV --no_mask_cor --orbit_dir $orbit_dir\
     --dem $dem_files_dir/Copernicus_DSM_COG_10_N16_00_E073_00_DEM.tif \
     --dem $dem_files_dir/Copernicus_DSM_COG_10_N17_00_E073_00_DEM.tif --ll info

if [[ -z "${NIGHTLY_GOLDEN_DIR}" ]]; then
  echo "no golden directory defined, no verification executed"
  exit 0
fi

set +e

echo "Validating $test_1_prod_path"
./alus_result_check.py -I $test_1_prod_path -G "$NIGHTLY_GOLDEN_DIR"/S1A_IW_SLC__1SDV_20210710T005537_20210710T005604_038708_049158_B74A_Orb_Stack_coh_deb_tc.tif -O SKIP_ALUs_VERSION
res1=$?

exit $(($res1))
