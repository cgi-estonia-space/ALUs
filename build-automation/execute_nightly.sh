#!/bin/bash

set -e

function print_help {
  echo "Usage:"
  echo "$0 <build ID - leave empty \"\" arg for local checks> <resources file> [aws profile - optional]"
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

if [[ -z "${build_id}" ]]; then
  echo "Performing locally"
else
  tar -xzvf ${build_id}.tar.gz
  # Alus binary and shared libs' location included in path.
  export PATH=$PATH:$PWD/${build_id}
fi

products_output=~/nightly_results
echo $products_output
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
*****Jupyter notebook tests*****"
python3 -m venv .env
source .env/bin/activate
./run_jupyter_tests.sh "$(pwd)/$build_id/jupyter-notebook" "$test_datasets_dir" "$(pwd)/$build_id" "$orbit_files_dir"
jupyter_test_exit=$?
deactivate

aws_profile=$3
if [[ -z "${aws_profile}" ]]; then
  echo "No AWS profile given, no uploading of results"
  exit 0
fi

set -e

aws s3api put-object --bucket alus-builds --key "alus-nightly-latest.tar.gz" --body ${build_id}.tar.gz --acl public-read --storage-class STANDARD_IA --profile $aws_profile
aws s3api put-object --bucket alus-builds --key "${build_id}/${build_id}.tar.gz" --body ${build_id}.tar.gz --acl public-read --storage-class STANDARD_IA --profile $aws_profile
echo "Uploaded binary package available at https://alus-builds.s3.eu-central-1.amazonaws.com/${build_id}/${build_id}.tar.gz"

cd $products_output
for file in *.tif; do
  aws s3api put-object --bucket alus-builds --key "${build_id}/${file}" --body $file --acl public-read --storage-class STANDARD_IA --profile $aws_profile
  echo "Uploaded resource available at https://alus-builds.s3.eu-central-1.amazonaws.com/${build_id}/${file}"
done

for file in *tc.tif; do
  png_file=${file%.*}.png
  gdal_translate -of PNG -ot Byte -scale $file $png_file
  aws s3api put-object --bucket alus-builds --key "${build_id}/${png_file}" --body $png_file --acl public-read --storage-class STANDARD_IA --profile $aws_profile
  echo "Uploaded resource available at https://alus-builds.s3.eu-central-1.amazonaws.com/${build_id}/${png_file}"
done

exit $(($disaster_test_exit | $virumaa_calibration_test_exit | $flood_test_exit | $maharashtra_calibration_test_exit | jupyter_test_exit))
