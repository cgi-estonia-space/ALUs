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

dem_files_argument="--dem $dem_files_dir/srtm_43_06.tif --dem $dem_files_dir/srtm_44_06.tif"
zipped_dem_files_argument="--dem $dem_files_dir/srtm_43_06.zip --dem $dem_files_dir/srtm_44_06.zip"

time alus --alg_name coherence-estimation-routine -i $test_dataset_dir/S1A_IW_SLC__1SDV_20200724T034334_20200724T034401_033591_03E49D_96AA.SAFE \
  -i $test_dataset_dir/S1A_IW_SLC__1SDV_20200805T034334_20200805T034401_033766_03E9F9_52F6.SAFE \
  -o $output_dir -p main_scene_identifier=S1A_IW_SLC__1SDV_20200724T034334,orbit_file_dir=$orbit_files_dir,subswath=IW1,polarization=VV \
  $dem_files_argument -x 3000 -y 3000
test_1_prod_path=$output_dir/S1A_IW_SLC__1SDV_20200724T034334_20200724T034401_033591_03E49D_96AA_Orb_Stack_IW1_coh_deb_tc.tif
mv $output_dir/S1A_IW_SLC__1SDV_20200724T034334_20200724T034401_033591_03E49D_96AA_Orb_Stack_coh_deb_tc.tif $test_1_prod_path

time alus --alg_name coherence-estimation-routine -i $test_dataset_dir/S1A_IW_SLC__1SDV_20200724T034334_20200724T034401_033591_03E49D_96AA.SAFE \
  -i $test_dataset_dir/S1A_IW_SLC__1SDV_20200805T034334_20200805T034401_033766_03E9F9_52F6.SAFE \
  -o $output_dir \
  -p main_scene_identifier=S1A_IW_SLC__1SDV_20200724T034334,orbit_file_dir=$orbit_files_dir,subswath=IW1,polarization=VV,main_scene_first_burst_index=6,main_scene_last_burst_index=6,secondary_scene_first_burst_index=5,secondary_scene_last_burst_index=7 \
  $dem_files_argument -x 3000 -y 3000
test_2_prod_path=$output_dir/S1A_IW_SLC__1SDV_20200724T034334_20200724T034401_033591_03E49D_96AA_Orb_Stack_IW1_b6_coh_deb_tc.tif
mv $output_dir/S1A_IW_SLC__1SDV_20200724T034334_20200724T034401_033591_03E49D_96AA_Orb_Stack_coh_deb_tc.tif $test_2_prod_path

time alus --alg_name coherence-estimation-routine -i $test_dataset_dir/S1A_IW_SLC__1SDV_20200724T034334_20200724T034401_033591_03E49D_96AA.SAFE \
  -i $test_dataset_dir/S1A_IW_SLC__1SDV_20200805T034334_20200805T034401_033766_03E9F9_52F6.SAFE -o $output_dir \
  -p main_scene_identifier=S1A_IW_SLC__1SDV_20200724T034334,orbit_file_dir=$orbit_files_dir,subswath=IW1,polarization=VV,main_scene_first_burst_index=4,main_scene_last_burst_index=7,secondary_scene_first_burst_index=3,secondary_scene_last_burst_index=8 \
  $dem_files_argument -x 3000 -y 3000
test_3_prod_path=$output_dir/S1A_IW_SLC__1SDV_20200724T034334_20200724T034401_033591_03E49D_96AA_Orb_Stack_IW1_b47_coh_deb_tc.tif
mv $output_dir/S1A_IW_SLC__1SDV_20200724T034334_20200724T034401_033591_03E49D_96AA_Orb_Stack_coh_deb_tc.tif $test_3_prod_path

time alus --alg_name coherence-estimation-routine -i $test_dataset_dir/S1A_IW_SLC__1SDV_20200724T034334_20200724T034401_033591_03E49D_96AA.SAFE \
  -i $test_dataset_dir/S1A_IW_SLC__1SDV_20200805T034334_20200805T034401_033766_03E9F9_52F6.SAFE \
  -o $output_dir \
  -p "main_scene_identifier=S1A_IW_SLC__1SDV_20200724T034334,orbit_file_dir=$orbit_files_dir,subswath=IW1,polarization=VV,aoi=POLYGON((35.47605514526368 33.919574396172536,35.46764373779297 33.87597405825278,35.56463241577149 33.87212589943945,35.57750701904297 33.92911789997693,35.47605514526368 33.919574396172536))" \
  $dem_files_argument -x 3000 -y 3000
test_4_prod_path=$output_dir/S1A_IW_SLC__1SDV_20200724T034334_20200724T034401_033591_03E49D_96AA_Orb_Stack_IW1_aoi_coh_deb_tc.tif
mv $output_dir/S1A_IW_SLC__1SDV_20200724T034334_20200724T034401_033591_03E49D_96AA_Orb_Stack_coh_deb_tc.tif $test_4_prod_path

time alus --alg_name coherence-estimation-routine -i $test_dataset_dir/S1A_IW_SLC__1SDV_20200724T034334_20200724T034401_033591_03E49D_96AA.zip \
  -i $test_dataset_dir/S1A_IW_SLC__1SDV_20200805T034334_20200805T034401_033766_03E9F9_52F6.zip -o $output_dir \
  -p "main_scene_identifier=S1A_IW_SLC__1SDV_20200724T034334,orbit_file_dir=$orbit_files_dir,subswath=IW1,polarization=VV,main_scene_first_burst_index=4,main_scene_last_burst_index=7,secondary_scene_first_burst_index=3,secondary_scene_last_burst_index=8" \
  $zipped_dem_files_argument -x 3000 -y 3000
test_5_prod_path=$output_dir/S1A_IW_SLC__1SDV_20200724T034334_20200724T034401_033591_03E49D_96AA_Orb_Stack_IW1_b47_coh_deb_tc_zipped.tif
mv $output_dir/S1A_IW_SLC__1SDV_20200724T034334_20200724T034401_033591_03E49D_96AA_Orb_Stack_coh_deb_tc.tif $test_5_prod_path

if [[ -z "${NIGHTLY_GOLDEN_DIR}" ]]; then
  echo "No golden directory defined, no verification executed"
  exit 0
fi

set +e

echo "Validating $test_1_prod_path"
./alus_result_check.py -I $test_1_prod_path -G "$NIGHTLY_GOLDEN_DIR"/S1A_IW_SLC__1SDV_20200724T034334_20200724T034401_033591_03E49D_96AA_Orb_Stack_IW1_coh_deb_tc.tif
res1=$?

echo "Validating $test_2_prod_path"
# This file will fail automatic gdal_convert scaling. Maybe GDAL bug, but the values range from 0...0.988 and conversion fails when converting to PNG.
./alus_result_check.py -I $test_2_prod_path -G "$NIGHTLY_GOLDEN_DIR"/S1A_IW_SLC__1SDV_20200724T034334_20200724T034401_033591_03E49D_96AA_Orb_Stack_IW1_b6_coh_deb_tc.tif
res2=$?

echo "Validating $test_3_prod_path"
./alus_result_check.py -I $test_3_prod_path -G "$NIGHTLY_GOLDEN_DIR"/S1A_IW_SLC__1SDV_20200724T034334_20200724T034401_033591_03E49D_96AA_Orb_Stack_IW1_b47_coh_deb_tc.tif
res3=$?

echo "Validating $test_4_prod_path"
./alus_result_check.py -I $test_4_prod_path -G "$NIGHTLY_GOLDEN_DIR"/S1A_IW_SLC__1SDV_20200724T034334_20200724T034401_033591_03E49D_96AA_Orb_Stack_IW1_aoi_coh_deb_tc.tif
res4=$?

echo "Validating $test_5_prod_path"
./alus_result_check.py -I $test_5_prod_path -G "$NIGHTLY_GOLDEN_DIR"/S1A_IW_SLC__1SDV_20200724T034334_20200724T034401_033591_03E49D_96AA_Orb_Stack_IW1_b47_coh_deb_tc.tif
res5=$?

exit $(($res1 | $res2 | $res3 | $res4 | $res5))
