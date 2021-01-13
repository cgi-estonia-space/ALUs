#!/bin/bash

set -e

function print_help {
    echo "Usage:"
    echo "$0 <build ID> <coherence test data folder> <dem files location> <calibration test data folder>"
}

if [[ "$#" != 4 ]]; then
    echo "Wrong count of input arguments"
    print_help
    exit 1
fi

build_id=$1

tar -xzvf ${build_id}.tar.gz

# Run scripts of different cases
mv run_beirut_disaster_test.sh ${build_id}
mv run_virumaa_calibration_test.sh ${build_id}

cd ${build_id}

# Run cases
./run_beirut_disaster_test.sh $2 $3
./run_virumaa_calibration_test.sh $4

cd ..

aws s3api put-object --bucket alus-builds --key "alus-nightly-latest.tar.gz" --body ${build_id}.tar.gz --acl public-read --storage-class STANDARD_IA --profile tarmo
aws s3api put-object --bucket alus-builds --key "${build_id}/${build_id}.tar.gz" --body ${build_id}.tar.gz --acl public-read --storage-class STANDARD_IA --profile tarmo
echo "Uploaded binary package available at https://alus-builds.s3.eu-central-1.amazonaws.com/${build_id}/${build_id}.tar.gz"

cd /tmp
for file in *.tif; do
        aws s3api put-object --bucket alus-builds --key "${build_id}/${file}" --body $file --acl public-read --storage-class STANDARD_IA --profile tarmo
        echo "Uploaded resource available at https://alus-builds.s3.eu-central-1.amazonaws.com/${build_id}/${file}"
done

for file in *tc.tif; do
        png_file=${file%.*}.png
        gdal_translate -of PNG -ot Byte -scale $file $png_file
        aws s3api put-object --bucket alus-builds --key "${build_id}/${png_file}" --body $png_file --acl public-read --storage-class STANDARD_IA --profile tarmo
        echo "Uploaded resource available at https://alus-builds.s3.eu-central-1.amazonaws.com/${build_id}/${png_file}"
done
