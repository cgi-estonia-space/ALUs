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

output_dir=$2/gfe
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

time alus-gfe -i $test_dataset_dir/T35VNE_20211102T093049_B02.jp2 -d $output_dir -f 6 -o 4 -p 50

if [[ -z "${NIGHTLY_GOLDEN_DIR}" ]]; then
  echo "No golden directory defined, no verification executed"
  exit 0
fi

set +e

ogrinfo -al $output_dir/T35VNE_20211102T093049_B02.sqlite > $output_dir/T35VNE_20211102T093049_B02_ogr_info.txt
diff -I '^INFO: Open of.*' $NIGHTLY_GOLDEN_DIR/T35VNE_20211102T093049_B02_ogr_info.txt \
$output_dir/T35VNE_20211102T093049_B02_ogr_info.txt

results_s2_ok=$?

exit $((results_s2_ok))
