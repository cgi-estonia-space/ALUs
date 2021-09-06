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
orbit_files_folder="$2/POEORB/"

dem_files_argument="--dem $dem_data_folder/srtm_43_06.tif --dem $dem_data_folder/srtm_44_06.tif"

time ./alus --alg_name coherence-estimation-routine -i $test_data_folder/../S1A_IW_SLC__1SDV_20200724T034334_20200724T034401_033591_03E49D_96AA.SAFE -i $test_data_folder/../S1A_IW_SLC__1SDV_20200805T034334_20200805T034401_033766_03E9F9_52F6.SAFE -o /tmp/ -p main_scene_identifier=S1A_IW_SLC__1SDV_20200724T034334,orbit_file_dir=$orbit_files_folder,subswath=IW1,polarization=VV $dem_files_argument -x 3000 -y 3000
gdalinfo -checksum /tmp/S1A_IW_SLC__1SDV_20200724T034334*coh*tc.tif
./alus_result_check.py -I /tmp/S1A_IW_SLC__1SDV_20200724T034334*coh*tc.tif -G "$NIGHTLY_GOLDEN_DIR"/S1A_IW_SLC__1SDV_20200724T034334_20200724T034401_033591_03E49D_96AA_Orb_Stack_coh_deb_tc.tif

beirut_single_burst_out="/tmp/beirut_single_burst"
mkdir $beirut_single_burst_out
time ./alus --alg_name coherence-estimation-routine -i $test_data_folder/../S1A_IW_SLC__1SDV_20200724T034334_20200724T034401_033591_03E49D_96AA.SAFE -i $test_data_folder/../S1A_IW_SLC__1SDV_20200805T034334_20200805T034401_033766_03E9F9_52F6.SAFE -o $beirut_single_burst_out -p main_scene_identifier=S1A_IW_SLC__1SDV_20200724T034334,orbit_file_dir=$orbit_files_folder,subswath=IW1,polarization=VV,main_scene_first_burst_index=6,main_scene_last_burst_index=6,secondary_scene_first_burst_index=5,secondary_scene_last_burst_index=7 $dem_files_argument -x 3000 -y 3000
mv $beirut_single_burst_out/*.tif /tmp/S1A_IW_SLC__1SDV_20200724T034334_beirut_single_burst_tc.tif
# This file will fail automatic gdal_convert scaling. Maybe GDAL bug, but the values range from 0...0.988 and conversion fails when converting to PNG.
gdalinfo -checksum -mm /tmp/S1A_IW_SLC__1SDV_20200724T034334_beirut_single_burst_tc.tif
./alus_result_check.py -I /tmp/S1A_IW_SLC__1SDV_20200724T034334_beirut_single_burst_tc.tif -G "$NIGHTLY_GOLDEN_DIR"/S1A_IW_SLC__1SDV_20200724T034334_beirut_single_burst_tc.tif

beirut_multiple_burst_out="/tmp/beirut_multiple_burst"
mkdir $beirut_multiple_burst_out
time ./alus --alg_name coherence-estimation-routine -i $test_data_folder/../S1A_IW_SLC__1SDV_20200724T034334_20200724T034401_033591_03E49D_96AA.SAFE -i $test_data_folder/../S1A_IW_SLC__1SDV_20200805T034334_20200805T034401_033766_03E9F9_52F6.SAFE -o $beirut_multiple_burst_out -p main_scene_identifier=S1A_IW_SLC__1SDV_20200724T034334,orbit_file_dir=$orbit_files_folder,subswath=IW1,polarization=VV,main_scene_first_burst_index=4,main_scene_last_burst_index=7,secondary_scene_first_burst_index=3,secondary_scene_last_burst_index=8 $dem_files_argument -x 3000 -y 3000
mv $beirut_multiple_burst_out/*.tif /tmp/S1A_IW_SLC__1SDV_20200724T034334_beirut_multiple_burst_tc.tif
gdalinfo -checksum -mm /tmp/S1A_IW_SLC__1SDV_20200724T034334_beirut_multiple_burst_tc.tif
./alus_result_check.py -I /tmp/S1A_IW_SLC__1SDV_20200724T034334_beirut_multiple_burst_tc.tif -G "$NIGHTLY_GOLDEN_DIR"/S1A_IW_SLC__1SDV_20200724T034334_beirut_multiple_burst_tc.tif

beirut_aoi_out="/tmp/beirut_aoi"
mkdir $beirut_aoi_out
time ./alus --alg_name coherence-estimation-routine -i $test_data_folder/../S1A_IW_SLC__1SDV_20200724T034334_20200724T034401_033591_03E49D_96AA.SAFE -i $test_data_folder/../S1A_IW_SLC__1SDV_20200805T034334_20200805T034401_033766_03E9F9_52F6.SAFE -o $beirut_aoi_out -p "main_scene_identifier=S1A_IW_SLC__1SDV_20200724T034334,orbit_file_dir=$orbit_files_folder,subswath=IW1,polarization=VV,aoi=POLYGON((35.47605514526368 33.919574396172536,35.46764373779297 33.87597405825278,35.56463241577149 33.87212589943945,35.57750701904297 33.92911789997693,35.47605514526368 33.919574396172536))" $dem_files_argument -x 3000 -y 3000
mv $beirut_aoi_out/*.tif /tmp/S1A_IW_SLC__1SDV_20200724T034334_beirut_aoi_tc.tif
gdalinfo -checksum -mm /tmp/S1A_IW_SLC__1SDV_20200724T034334_beirut_aoi_tc.tif
./alus_result_check.py -I /tmp/S1A_IW_SLC__1SDV_20200724T034334_beirut_aoi_tc.tif -G "$NIGHTLY_GOLDEN_DIR"/S1A_IW_SLC__1SDV_20200724T034334_beirut_aoi_tc.tif
