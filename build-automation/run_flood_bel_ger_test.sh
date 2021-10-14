#!/bin/bash

set -e

function print_help {
    echo "Usage:"
    echo "$0 <test data folder> <dem files location> <orbit files location> [optional - output folder]"
}

if [ $# -lt 3 ]; then
    echo "Wrong count of input arguments"
    print_help
    exit 1
fi

test_dataset_dir=$1
dem_files_dir=$2
orbit_files_dir=$3

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

dem_files_arg="--dem $dem_files_dir/srtm_37_02.tif --dem $dem_files_dir/srtm_37_03.tif --dem $dem_files_dir/srtm_38_02.tif --dem $dem_files_dir/srtm_38_03.tif "

test_1_prod_path=$output_dir/S1A_IW_SLC__1SDV_20210703T055050_20210703T055117_038609_048E45_35F7_Orb_Stack_coh_deb_tc.tif
time alus --alg_name coherence-estimation-routine $dem_files_arg \
     -i $test_dataset_dir/S1B_IW_SLC__1SDV_20210721T055001_20210721T055028_027888_0353E2_E1B5.SAFE -i $test_dataset_dir/S1A_IW_SLC__1SDV_20210703T055050_20210703T055117_038609_048E45_35F7.SAFE \
     -o $output_dir -x 2000 -y 2000 \
     -p "main_scene_identifier=S1A_IW_SL,secondary_scene_orbit_file=$orbit_files_dir/S1B_OPER_AUX_POEORB_OPOD_20210810T111942_V20210720T225942_20210722T005942.EOF,main_scene_orbit_file=$orbit_files_dir/S1A_OPER_AUX_POEORB_OPOD_20210723T121923_V20210702T225942_20210704T005942.EOF,subswath=IW3,aoi=POLYGON ((3.76064043932478 50.6679002753201,4.81930157970497 50.5884971985178,4.65806260842462 50.0309601054367,3.65031903792243 50.1622939049033,3.76064043932478 50.6679002753201)),polarization=VV"

test_2_prod_path=$output_dir/S1B_IW_SLC__1SDV_20210615T054959_20210615T055026_027363_0344A0_83FE_Orb_Stack_coh_deb_tc.tif
time alus --alg_name coherence-estimation-routine $dem_files_arg \
     -i $test_dataset_dir/S1B_IW_SLC__1SDV_20210615T054959_20210615T055026_027363_0344A0_83FE.SAFE -i $test_dataset_dir/S1A_IW_SLC__1SDV_20210703T055050_20210703T055117_038609_048E45_35F7.SAFE \
     -o $output_dir -x 2000 -y 2000 \
     -p "main_scene_identifier=S1B_IW_SLC__1SDV_20210615T054959,main_scene_orbit_file=$orbit_files_dir/S1B_OPER_AUX_POEORB_OPOD_20210705T111814_V20210614T225942_20210616T005942.EOF,secondary_scene_orbit_file=$orbit_files_dir/S1A_OPER_AUX_POEORB_OPOD_20210723T121923_V20210702T225942_20210704T005942.EOF,subswath=IW3,polarization=VV,aoi=POLYGON ((3.76064043932478 50.6679002753201,4.81930157970497 50.5884971985178,4.65806260842462 50.0309601054367,3.65031903792243 50.1622939049033,3.76064043932478 50.6679002753201))"

if [[ -z "${NIGHTLY_GOLDEN_DIR}" ]]; then
  echo "no golden directory defined, no verification executed"
  exit 0
fi

set +e

echo "Validating $test_1_prod_path"
./alus_result_check.py -I $test_1_prod_path -G "$NIGHTLY_GOLDEN_DIR"/S1A_IW_SLC__1SDV_20210703T055050_20210703T055117_038609_048E45_35F7_Orb_Stack_coh_deb_tc.tif
res1=$?
echo "Validating $test_2_prod_path"
./alus_result_check.py -I $test_2_prod_path -G "$NIGHTLY_GOLDEN_DIR"/S1B_IW_SLC__1SDV_20210615T054959_20210615T055026_027363_0344A0_83FE_Orb_Stack_coh_deb_tc.tif
res2=$?

exit $(($res1 | $res2))
