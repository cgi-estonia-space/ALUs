#!/bin/bash

set -e

function print_help {
  echo "Usage:"
  echo "$0 <build ID - leave empty \"\" arg for local checks> <resources file>"
}

if [ $# -lt 2 ]; then
  echo "Wrong count of input arguments"
  print_help
  exit 1
fi

build_id=$1
nightly_resources=$2
test_datasets_dir=$(grep test_data $nightly_resources | awk -F'[=]' '{print $2}')
dem_files_dir=$(grep dem_files $nightly_resources | awk -F'[=]' '{print $2}')
orbit_files_dir=$(grep orbit_files $nightly_resources | awk -F'[=]' '{print $2}')
products_output=$(grep results $nightly_resources | awk -F'[=]' '{print $2}')

if [[ -z "${build_id}" ]]; then
  echo "Performing locally"
else
  tar -xzvf ${build_id}.tar.gz
  # Alus binary location included in path.
  export PATH=$PATH:$PWD
fi

mkdir -p $products_output
rm -rf $products_output/*

set +e

# Run cases
echo "
*****Beirut disaster coherence scenes*****"
./run_beirut_disaster_test.sh "$test_datasets_dir" "$dem_files_dir" "$orbit_files_dir" $products_output
disaster_test_exit=$?
echo "
*****BEL and GER flood coherence scenes*****"
./run_flood_bel_ger_test.sh $test_datasets_dir $dem_files_dir $orbit_files_dir $products_output
flood_test_exit=$?
echo "
*****Virumaa calibration scene*****"
./run_virumaa_calibration_chain_test.sh $test_datasets_dir $dem_files_dir $products_output
virumaa_calibration_test_exit=$?
echo "
*****Maharashtra flood calibration scene*****"
./run_maharashtra_calibration_test.sh $test_datasets_dir $dem_files_dir $products_output
maharashtra_calibration_test_exit=$?
echo "
*****Resampling tests******"
./run_resample_test.sh $test_datasets_dir $products_output
resample_test_exit=$?
echo "
*****Gabor feature extraction tests*****"
./run_gabor_feature_extraction_test.sh $test_datasets_dir $products_output
gabor_feature_extraction_exit=$?
echo "
*****Jupyter notebook tests*****"
python3 -m venv .env
source .env/bin/activate
./run_jupyter_tests.sh "$(pwd)/jupyter-notebook" "$test_datasets_dir" "$(pwd)" "$orbit_files_dir" $products_output
jupyter_test_exit=$?
deactivate

exit $((disaster_test_exit | virumaa_calibration_test_exit | flood_test_exit | maharashtra_calibration_test_exit | resample_test_exit | gabor_feature_extraction_exit | jupyter_test_exit))
