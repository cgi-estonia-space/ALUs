#!/bin/bash


function print_help {
    echo "Usage:"
    echo "./dem_supply.sh <metadata file (SAFE folder/zip or .dimap)>"
    echo "For example:"
    echo "./dem_supply.sh S1A....Orb.dim"
    echo "./dem_supply.sh S1A_1342...865.SAFE"
    echo "./dem_supply.sh S1A_1342...876.zip"
}

if [[ "$#" != 1 ]]; then
    echo "Wrong count of input arguments"
    print_help
    exit 1
fi

metadata_file=$1

set -e

# Check if 'eio' tool is installed correctly on system, prints errors if not okay
eio selfcheck > /dev/null

# Extract '.dim' or '.zip' or 'SAFE'
type_string="${metadata_file: -4}"

# Bounds for 'eio'
left=""
bottom=""
right=""
top=""

# Parse coordinates from a string like '59.125515,23.795147 59.543510,28.198971 57.934528,28.672609 57.523579,24.467125'
# into variables above
function parse_safe_coordinates {
    coordinates=$1
    echo "parsing $coordinates"
    coordinates_split=($coordinates)
    left=$(echo ${coordinates_split[0]} | awk -F',' '{print $2}')
    bottom=$(echo ${coordinates_split[3]} | awk -F',' '{print $1}')
    right=$(echo ${coordinates_split[2]} | awk -F',' '{print $2}')
    top=$(echo ${coordinates_split[1]} | awk -F',' '{print $1}')
}

if [[ "$type_string" == "SAFE" || "$type_string" == "AFE/" ]]; then
# SAFE format has coordinates in xml tag as '<gml:coordinates>.....</gml:coordinates>', also remove tags and whitespace
    coordinates=$(grep gml:coordinates $metadata_file/manifest.safe | sed -e 's/<[^>]*>//g' | xargs)
    parse_safe_coordinates "$coordinates"
elif [[ "$type_string" == ".zip" ]]; then
# Get the list of contents and only extract 'manifest.safe' file path inside archive
    manifest_from_list=$(unzip -l $metadata_file | grep manifest.safe)
    manifest_from_list_arr=($manifest_from_list)
    manifest_path=${manifest_from_list_arr[-1]}
# SAFE format has coordinates in xml tag as '<gml:coordinates>.....</gml:coordinates>', also remove tags and whitespace
    coordinates=$(unzip -p $metadata_file $manifest_path | grep gml:coordinates | sed -e 's/<[^>]*>//g' | xargs)
    parse_safe_coordinates "$coordinates"
elif [[ "$type_string" == ".dim" ]]; then
    left=$(grep last_near_long ${metadata_file} | head -1 | sed -e 's/<[^>]*>//g' | xargs)
    bottom=$(grep first_near_lat ${metadata_file} | head -1 | sed -e 's/<[^>]*>//g' | xargs)
    right=$(grep first_far_long ${metadata_file} | head -1 | sed -e 's/<[^>]*>//g' | xargs)
    top=$(grep last_far_lat ${metadata_file}  | head -1 | sed -e 's/<[^>]*>//g' | xargs)
else
    echo "Wrong metadata supplied."
    print_help
    exit 2
fi

margin="0.5"
left_with_margin=$(echo "$left - $margin" | bc)
bottom_with_margin=$(echo "$bottom - $margin" | bc)
right_with_margin=$(echo "$right + $margin" | bc)
top_with_margin=$(echo "$top + $margin" | bc)

top_limit=60.0
clip_top=`echo "$top_with_margin >= $top_limit" | bc`
if [ $clip_top -eq 1 ]; then
  top_with_margin=59.99999999
fi

bottom_limit=-60.0
clip_bottom=`echo "$bottom_with_margin <= $bottom_limit" | bc`
if [ $clip_bottom -eq 1 ]; then
  bottom_with_margin=-59.9999999
fi

dem_dir="/tmp/elevation"
log_file=${dem_dir}/log.txt
echo "eio --product SRTM3 --cache_dir $dem_dir seed --bounds $left_with_margin $bottom_with_margin $right_with_margin $top_with_margin"
eio --product SRTM3 --cache_dir $dem_dir seed --bounds $left_with_margin $bottom_with_margin $right_with_margin $top_with_margin | tee $log_file

# Parse DEM files for this scene
dem_files_raw=$(grep "SRTM3 associated" $log_file | head -1 | cut -c18-)
dem_files_associated=${dem_files_raw//SRTM3 associated/}

echo "Log saved to $log_file" 
echo "DEM files for scene:"
echo $dem_files_associated
echo $dem_files_associated >> $log_file
