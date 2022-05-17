#!/bin/bash

set -e

function print_help {
  echo "Usage:"
  echo "$0 <test data folder> [optional - output folder]"
}

if [ $# -lt 1 ]; then
  echo "Wrong count of input arguments"
  print_help
  exit 1
fi

test_dataset_dir=$1

output_dir=$2
if [ $# -eq 1 ]; then
  me=$0
  # foo.ext
  me=${me##*/}
  # foo
  me=${me%.*}
  output_dir="/tmp/$me"
  echo "Created output folder for results - $output_dir"
fi

mkdir -p $output_dir

time alus-resa -i $test_dataset_dir/S2B_MSIL1C_20211102T093049_N0301_R136_T35VNE_20211102T114211.zip \
-d $output_dir --dim_band 2 --tile_dim 4096x4096 --overlap 128 -m lanczos --exclude 1 --exclude 2 --exclude 3 --exclude 4 \
--exclude 5 --exclude 6 --exclude 7 --exclude 8 --exclude 10 --exclude 11 --exclude 12 --exclude 13 -f Gtiff

time alus-resa -i $test_dataset_dir/srtm_37_02.tif -d $output_dir --width 12000 --height 12000 \
--tile_dim 6000x6000 -m nearest-neighbour

if [[ -z "${NIGHTLY_GOLDEN_DIR}" ]]; then
  echo "No golden directory defined, no verification executed"
  exit 0
fi

set +e

results_s2=($output_dir/T35VNE_20211102T093049_B8A_*)
results_s2_length=${#results_s2[@]}
results_s2_ok=1
if [ $results_s2_length -ne 9 ];
  then
    echo "Expected 9 S2 tiles, actual - $results_s2_length"
    results_s2_ok=1
  else
    results_s2_ok=0
    for result in "${results_s2[@]}"
    do
      ./alus_result_check.py -I $result -G $NIGHTLY_GOLDEN_DIR/${result##*/} -O SKIP_ALUs_VERSION
      if [ $? -ne 0 ];
        then
          results_s2_ok=1
          echo "$result not correct"
          break
      fi
    done
fi

results_generic=($output_dir/srtm_37_02_*)
results_generic_length=${#results_generic[@]}
results_generic_ok=1
if [ $results_generic_length -ne 4 ];
  then
    echo "Expected 4 SRTM3 tiles, actual - $results_generic_length"
    results_generic_ok=1
  else
    results_generic_ok=0
    for result in "${results_generic[@]}"
    do
      ./alus_result_check.py -I $result -G $NIGHTLY_GOLDEN_DIR/${result##*/} -O SKIP_ALUs_VERSION
      if [ $? -ne 0 ];
        then
          results_generic_ok=1
          echo "$result not correct"
          break
      fi
    done
fi

exit $((results_s2_ok | results_generic_ok))
