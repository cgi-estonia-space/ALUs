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


time ./alus --alg_name coherence-estimation-routine -i $test_data_folder/../S1A_IW_SLC__1SDV_20200724T034334_20200724T034401_033591_03E49D_96AA.SAFE -i $test_data_folder/../S1A_IW_SLC__1SDV_20200805T034334_20200805T034401_033766_03E9F9_52F6.SAFE -o /tmp/ -p main_scene_identifier=S1A_IW_SLC__1SDV_20200724T034334,orbit_file_dir=$orbit_files_folder,subswath=IW1,polarization=VV --dem $dem_data_folder/srtm_43_06.tif --dem $dem_data_folder/srtm_44_06.tif -x 3000 -y 3000
gdalinfo -checksum /tmp/S1A_IW_SLC__1SDV_20200724T034334*coh*tc.tif
./alus_result_check.py -I /tmp/S1A_IW_SLC__1SDV_20200724T034334*coh*tc.tif -G "$NIGHTLY_GOLDEN_DIR"/S1A_IW_SLC__1SDV_20200724T034334_20200724T034401_033591_03E49D_96AA_Orb_Stack_coh_deb_tc.tif
