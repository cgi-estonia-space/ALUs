#!/usr/bin/env bash

# NB! Should be executed after activating python virtual environment
jupyter_folder=$1
test_dataset_dir=$2
exec_dir=$3
orbit_files_dir=$4

output_dir=$5
if [ $# -eq 4 ]; then
  me=$0
  # foo.ext
  me=${me##*/}
  # foo
  me=${me%.*}

  mkdir -p "/tmp/$me"
  output_dir="/tmp/$me"
  echo "Created output directory for results - $output_dir"
fi

if [ "$jupyter_folder" == "" ]; then
  echo "Jupyter Notebook directory not provided. Please enter a directory as the script's first argument."
  exit 1
fi

if [ "$test_dataset_dir" == "" ]; then
  echo "Test dataset directory not provided. Please enter a directory as the script's second argument."
  exit 1
fi

if [ "$exec_dir" == "" ]; then
  echo "ALUs executable directory not provided. Please enter a directory as the script's third argument."
  exit 1
fi

if [ "$orbit_files_dir" == "" ]; then
  echo "Orbit files directory not provided. Please enter a directory as the script's fourth argument."
  exit 1
fi

python -m pip install wheel
python -m pip install -r "$jupyter_folder"/requirements.txt
original_folder=$(pwd)
cd "$jupyter_folder" || exit 1
python -m pytest -v -s --junitxml="BITBUCKET_CLONE_DIR"/build/test-reports/pytest-report.xml regression-tests --exec_dir "$exec_dir" \
  --cal_output_file "$output_dir" --calib_input "$test_dataset_dir/S1A_IW_SLC__1SDV_20210722T005537_20210722T005604_038883_049695_2E58.zip" \
  --orbit_dir "$orbit_files_dir" --coh_reference "$test_dataset_dir/S1A_IW_SLC__1SDV_20200724T034334_20200724T034401_033591_03E49D_96AA.zip" \
  --coh_secondary "$test_dataset_dir/S1A_IW_SLC__1SDV_20200805T034334_20200805T034401_033766_03E9F9_52F6.zip" \
  --coh_output_file "$output_dir"
res1=$?

if [[ -z "${NIGHTLY_GOLDEN_DIR}" ]]; then
  echo "no golden directory defined, no verification executed"
  exit 0
fi

calibration_output_file="$output_dir/S1A_IW_SLC__1SDV_20210722T005537_20210722T005604_038883_049695_2E58_Cal_IW2_deb_tc.tif"
echo "Validating $calibration_output_file"
ls -lah ..
"$original_folder"/alus_result_check.py -I "$calibration_output_file" -G "$NIGHTLY_GOLDEN_DIR"/S1A_IW_SLC__1SDV_20210722T005537_20210722T005604_038883_049695_2E58_Cal_IW2_deb_tc_jupyter.tif
res2=$?

coherence_output_file="$output_dir/S1A_IW_SLC__1SDV_20200724T034334_20200724T034401_033591_03E49D_96AA_Orb_Stack_coh_deb_tc.tif"
echo "Validating $coherence_output_file"
"$original_folder"/alus_result_check.py -I "$coherence_output_file" -G "$NIGHTLY_GOLDEN_DIR"/S1A_IW_SLC__1SDV_20200724T034334_20200724T034401_033591_03E49D_96AA_Orb_Stack_coh_deb_TC_jupyter.tif
res3=$?

exit $((res1 | res2 | res3))
