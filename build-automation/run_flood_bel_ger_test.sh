#!/bin/bash

set -e

function print_help {
    echo "Usage:"
    echo "$0 <test data folder> <srtm3 dem files location> <COPDEM 30m files location> <orbit files location> [optional - output folder]"
}

if [ $# -lt 4 ]; then
    echo "Wrong count of input arguments"
    print_help
    exit 1
fi

test_dataset_dir=$1
srtm3_files_dir=$2
copdem30m_files_dir=$3
orbit_files_dir=$4

output_dir=$5
if [ $# -eq 4 ]; then
  me=$0
  # foo.ext
  me=${me##*/}
  # foo
  me=${me%.*}
  output_dir="/tmp/$me"
  echo "Created output folder for results - $output_dir"
fi

mkdir -p $output_dir

srtm3_files_arg="--dem $srtm3_files_dir/srtm_37_02.tif --dem $srtm3_files_dir/srtm_37_03.tif --dem $srtm3_files_dir/srtm_38_02.tif --dem $srtm3_files_dir/srtm_38_03.tif "
zipped_srtm3_files_arg="--dem $srtm3_files_dir/srtm_37_02.zip --dem $srtm3_files_dir/srtm_37_03.zip --dem $srtm3_files_dir/srtm_38_02.zip --dem $srtm3_files_dir/srtm_38_03.zip "

test_1_prod_path=$output_dir/S1A_IW_SLC__1SDV_20210703T055050_20210703T055117_038609_048E45_35F7_Orb_Stack_coh_deb_tc.tif
time alus-coh -r $test_dataset_dir/S1A_IW_SLC__1SDV_20210703T055050_20210703T055117_038609_048E45_35F7.SAFE -s $test_dataset_dir/S1B_IW_SLC__1SDV_20210721T055001_20210721T055028_027888_0353E2_E1B5.SAFE \
     -o $output_dir --sw IW3 -p VV --orbit_ref $orbit_files_dir/S1A_OPER_AUX_POEORB_OPOD_20210723T121923_V20210702T225942_20210704T005942.EOF.zip \
     --orbit_sec $orbit_files_dir/S1B_OPER_AUX_POEORB_OPOD_20210810T111942_V20210720T225942_20210722T005942.EOF.zip \
     -a "POLYGON ((3.76064043932478 50.6679002753201,4.81930157970497 50.5884971985178,4.65806260842462 50.0309601054367,3.65031903792243 50.1622939049033,3.76064043932478 50.6679002753201))" \
     $srtm3_files_arg --ll info

test_2_prod_path=$output_dir/S1B_IW_SLC__1SDV_20210615T054959_20210615T055026_027363_0344A0_83FE_Orb_Stack_coh_deb_tc.tif
time alus-coh -r $test_dataset_dir/S1B_IW_SLC__1SDV_20210615T054959_20210615T055026_027363_0344A0_83FE.SAFE -s $test_dataset_dir/S1A_IW_SLC__1SDV_20210703T055050_20210703T055117_038609_048E45_35F7.SAFE \
     --orbit_ref $orbit_files_dir/S1B_OPER_AUX_POEORB_OPOD_20210705T111814_V20210614T225942_20210616T005942.EOF.zip --orbit_sec $orbit_files_dir/S1A_OPER_AUX_POEORB_OPOD_20210723T121923_V20210702T225942_20210704T005942.EOF.zip \
     --sw IW3 -p VV --aoi "POLYGON ((3.76064043932478 50.6679002753201,4.81930157970497 50.5884971985178,4.65806260842462 50.0309601054367,3.65031903792243 50.1622939049033,3.76064043932478 50.6679002753201))" \
     $srtm3_files_arg -o $output_dir --ll info

test_3_prod_path=$output_dir/S1B_IW_SLC__1SDV_20210615T054959_20210615T055026_027363_0344A0_83FE_Orb_Stack_coh_deb_tc_zipped.tif
time alus-coh -r $test_dataset_dir/S1B_IW_SLC__1SDV_20210615T054959_20210615T055026_027363_0344A0_83FE.zip -s $test_dataset_dir/S1A_IW_SLC__1SDV_20210703T055050_20210703T055117_038609_048E45_35F7.zip \
     --orbit_ref $orbit_files_dir/S1B_OPER_AUX_POEORB_OPOD_20210705T111814_V20210614T225942_20210616T005942.EOF.zip \
     --orbit_sec $orbit_files_dir/S1A_OPER_AUX_POEORB_OPOD_20210723T121923_V20210702T225942_20210704T005942.EOF.zip \
     --sw IW3 -p VV --aoi "POLYGON ((3.76064043932478 50.6679002753201,4.81930157970497 50.5884971985178,4.65806260842462 50.0309601054367,3.65031903792243 50.1622939049033,3.76064043932478 50.6679002753201))" \
     $zipped_srtm3_files_arg -o $test_3_prod_path --ll info

test_4_prod_path=$output_dir/S1A_IW_SLC__1SDV_20210703T055050_20210703T055117_038609_048E45_35F7_Orb_Stack_coh_deb_tc_IW_1_2_3_mrg.tif
time alus-coh -r $test_dataset_dir/S1A_IW_SLC__1SDV_20210703T055050_20210703T055117_038609_048E45_35F7.SAFE -s $test_dataset_dir/S1B_IW_SLC__1SDV_20210721T055001_20210721T055028_027888_0353E2_E1B5.SAFE \
     --orbit_ref $orbit_files_dir/S1A_OPER_AUX_POEORB_OPOD_20210723T121923_V20210702T225942_20210704T005942.EOF.zip \
     --orbit_sec $orbit_files_dir/S1B_OPER_AUX_POEORB_OPOD_20210810T111942_V20210720T225942_20210722T005942.EOF.zip \
     -p VV --aoi "POLYGON((4.2352294921875 50.61113171332363,6.306152343750001 50.54834449067479,6.273193359375 50.27178780378986,4.240722656250001 50.34896578114507,4.2352294921875 50.61113171332363))" \
     $srtm3_files_arg -o $test_4_prod_path --ll info

test_5_prod_path=$output_dir/S1A_IW_SLC__1SDV_20210703T055050_20210703T055117_038609_048E45_35F7_Orb_Stack_coh_deb_tc_IW_2_3_mrg.tif
time alus-coh -r $test_dataset_dir/S1A_IW_SLC__1SDV_20210703T055050_20210703T055117_038609_048E45_35F7.SAFE -s $test_dataset_dir/S1B_IW_SLC__1SDV_20210721T055001_20210721T055028_027888_0353E2_E1B5.SAFE \
     --orbit_ref $orbit_files_dir/S1A_OPER_AUX_POEORB_OPOD_20210723T121923_V20210702T225942_20210704T005942.EOF.zip \
     --orbit_sec $orbit_files_dir/S1B_OPER_AUX_POEORB_OPOD_20210810T111942_V20210720T225942_20210722T005942.EOF.zip \
     -p VV --aoi "POLYGON((4.46044921875 50.22963791789675,4.993286132812499 50.21909462044749,4.987792968749999 50.071243660444736,4.449462890625 50.10296448723352,4.46044921875 50.22963791789675))" \
     $srtm3_files_arg -o $test_5_prod_path --ll info

# One without orbit files.
test_6_prod_path=$output_dir/S1A_IW_SLC__1SDV_20210703T055050_20210703T055117_038609_048E45_35F7_Orb_Stack_coh_deb_tc_mrg_no_orb.tif
time alus-coh -r $test_dataset_dir/S1A_IW_SLC__1SDV_20210703T055050_20210703T055117_038609_048E45_35F7.SAFE -s $test_dataset_dir/S1B_IW_SLC__1SDV_20210721T055001_20210721T055028_027888_0353E2_E1B5.SAFE \
     -p VH $srtm3_files_arg -o $test_6_prod_path --ll info

test_7_prod_path=$output_dir/S1A_IW_SLC__1SDV_20210703T055050_20210703T055117_038609_048E45_35F7_Orb_Stack_coh_deb_tc_IW_2_3_mrg_shp.tif
time alus-coh -r $test_dataset_dir/S1A_IW_SLC__1SDV_20210703T055050_20210703T055117_038609_048E45_35F7.SAFE -s $test_dataset_dir/S1B_IW_SLC__1SDV_20210721T055001_20210721T055028_027888_0353E2_E1B5.SAFE \
     --orbit_ref $orbit_files_dir/S1A_OPER_AUX_POEORB_OPOD_20210723T121923_V20210702T225942_20210704T005942.EOF.zip \
     --orbit_sec $orbit_files_dir/S1B_OPER_AUX_POEORB_OPOD_20210810T111942_V20210720T225942_20210722T005942.EOF.zip \
     -p VV --aoi $test_dataset_dir/flood_bel_ger_test7_aoi.shp $srtm3_files_arg -o $test_7_prod_path --ll info

# Copernicus DEM 30m.
test_8_prod_path=$output_dir/S1B_IW_SLC__1SDV_20210615T054959_20210615T055026_027363_0344A0_83FE_Orb_IW3_b18_Stack_coh_deb_tc_copdem.tif
time alus-coh -r $test_dataset_dir/S1B_IW_SLC__1SDV_20210615T054959_20210615T055026_027363_0344A0_83FE.SAFE \
     -s $test_dataset_dir/S1B_IW_SLC__1SDV_20210721T055001_20210721T055028_027888_0353E2_E1B5.SAFE \
     --orbit_ref $orbit_files_dir/S1B_OPER_AUX_POEORB_OPOD_20210705T111814_V20210614T225942_20210616T005942.EOF.zip \
     --orbit_sec $orbit_files_dir/S1B_OPER_AUX_POEORB_OPOD_20210810T111942_V20210720T225942_20210722T005942.EOF.zip \
     -p VH --sw IW3 --b_ref1 1 --b_ref2 8 --b_sec1 1 --b_sec2 8  --ll info -o $test_8_prod_path \
     --dem $copdem30m_files_dir/Copernicus_DSM_COG_10_N51_00_E003_00_DEM.tif \
     --dem $copdem30m_files_dir/Copernicus_DSM_COG_10_N51_00_E004_00_DEM.tif \
     --dem $copdem30m_files_dir/Copernicus_DSM_COG_10_N51_00_E005_00_DEM.tif \
     --dem $copdem30m_files_dir/Copernicus_DSM_COG_10_N50_00_E003_00_DEM.tif \
     --dem $copdem30m_files_dir/Copernicus_DSM_COG_10_N50_00_E004_00_DEM.tif


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
echo "Validating $test_3_prod_path"
./alus_result_check.py -I $test_3_prod_path -G "$NIGHTLY_GOLDEN_DIR"/S1B_IW_SLC__1SDV_20210615T054959_20210615T055026_027363_0344A0_83FE_Orb_Stack_coh_deb_tc.tif
res3=$?
echo "Validating $test_4_prod_path"
./alus_result_check.py -I $test_4_prod_path -G "$NIGHTLY_GOLDEN_DIR"/S1A_IW_SLC__1SDV_20210703T055050_20210703T055117_038609_048E45_35F7_Orb_Stack_coh_deb_tc_IW_1_2_3_mrg.tif
res4=$?
echo "Validating $test_5_prod_path"
./alus_result_check.py -I $test_5_prod_path -G "$NIGHTLY_GOLDEN_DIR"/S1A_IW_SLC__1SDV_20210703T055050_20210703T055117_038609_048E45_35F7_Orb_Stack_coh_deb_tc_IW_2_3_mrg.tif
res5=$?
echo "Validating $test_6_prod_path"
./alus_result_check.py -I $test_6_prod_path -G "$NIGHTLY_GOLDEN_DIR"/S1A_IW_SLC__1SDV_20210703T055050_20210703T055117_038609_048E45_35F7_Orb_Stack_coh_deb_tc_mrg_no_orb.tif
res6=$?
echo "Validating $test_7_prod_path"
./alus_result_check.py -I $test_7_prod_path -G "$NIGHTLY_GOLDEN_DIR"/S1A_IW_SLC__1SDV_20210703T055050_20210703T055117_038609_048E45_35F7_Orb_Stack_coh_deb_tc_IW_2_3_mrg.tif
res7=$?
echo "Validating $test_8_prod_path"
./alus_result_check.py -I $test_8_prod_path -G "$NIGHTLY_GOLDEN_DIR"/S1B_IW_SLC__1SDV_20210615T054959_20210615T055026_027363_0344A0_83FE_Orb_IW3_b18_Stack_coh_deb_tc_copdem.tif
res8=$?

exit $(($res1 | $res2 | $res3 | $res4 | $res5 | $res6 | $res7 | $res8))
