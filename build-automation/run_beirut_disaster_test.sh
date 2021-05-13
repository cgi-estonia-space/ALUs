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
dem_data_folder=$2

time ./alus --alg_name coherence-estimation-routine -i $test_data_folder/S1B_IW_SLC__1SDV_20200730T034254_20200730T034321_022695_02B131_E8DD_split_Orb.tif -i $test_data_folder/S1A_IW_SLC__1SDV_20200805T034334_20200805T034401_033766_03E9F9_52F6_split_Orb.tif --aux $test_data_folder/S1B_IW_SLC__1SDV_20200730T034254_20200730T034321_022695_02B131_E8DD_split_Orb.dim --aux $test_data_folder/S1A_IW_SLC__1SDV_20200805T034334_20200805T034401_033766_03E9F9_52F6_split_Orb.dim --aux $test_data_folder/S1B_IW_SLC__1SDV_20200730T034254_20200730T034321_022695_02B131_E8DD_Orb_Stack_deb.dim -o /tmp/alus_beirut_S1B_IW_SLC__1SDV_20200730T034254_20200730T034321_022695_02B131_E8DD_split_Orb.tif -p "master=S1B_IW_SLC__1SDV_20200730T034254_20200730T034321_022695_02B131_E8DD_split_Orb,master_metadata=S1B_IW_SLC__1SDV_20200730T034254_20200730T034321_022695_02B131_E8DD_split_Orb,coherence_terrain_correction_metadata=S1B_IW_SLC__1SDV_20200730T034254_20200730T034321_022695_02B131_E8DD_Orb_Stack_deb,per_process_gpu_memory_fraction=0.15,subtract_flat_earth_phase=true" --dem $dem_data_folder/srtm_43_06.tif --dem $dem_data_folder/srtm_44_06.tif -x 2000 -y 2000

time ./alus --alg_name coherence-estimation-routine -i $test_data_folder/S1B_IW_SLC__1SDV_20200730T034254_20200730T034321_022695_02B131_E8DD_split_burst_5-7_Orb.tif -i $test_data_folder/S1A_IW_SLC__1SDV_20200805T034334_20200805T034401_033766_03E9F9_52F6_split_burst_6-8_Orb.tif --aux $test_data_folder/S1B_IW_SLC__1SDV_20200730T034254_20200730T034321_022695_02B131_E8DD_split_burst_5-7_Orb.dim --aux $test_data_folder/S1A_IW_SLC__1SDV_20200805T034334_20200805T034401_033766_03E9F9_52F6_split_burst_6-8_Orb.dim --aux $test_data_folder/S1B_IW_SLC__1SDV_20200730T034254_20200730T034321_022695_02B131_E8DD_Orb_Stack_burst_5-8.dim -o /tmp/alus_beirut_S1B_IW_SLC__1SDV_20200730T034254_20200730T034321_022695_02B131_E8DD_split_Orb_burst_5-8.tif -p "master=burst_5-7,master_metadata=burst_5-7,coherence_terrain_correction_metadata=burst_5-8,per_process_gpu_memory_fraction=0.3,subtract_flat_earth_phase=true" --dem $dem_data_folder/srtm_43_06.tif --dem $dem_data_folder/srtm_44_06.tif -x 2000 -y 2000

