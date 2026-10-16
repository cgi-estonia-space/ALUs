#!/bin/bash

set -euo pipefail

function print_help {
    echo "Usage:"
    echo "$0 <SLC dir> <ETAD dir> <COPDEM 30m COG dir> <orbit files dir> [output dir]"
    echo "Environment: NIGHTLY_GOLDEN_DIR=<SNAP golden dir>"
}

if [[ $# -lt 4 || $# -gt 5 ]]; then
    echo "Wrong count of input arguments" >&2
    print_help
    exit 1
fi

slc_files_dir=$1
etad_files_dir=$2
dem_files_dir=$3
orbit_files_dir=$4
if [[ $# -eq 5 ]]; then
    output_dir=${5%/}
else
    script_name=${0##*/}
    output_dir="/tmp/${script_name%.*}"
fi

first_burst=1
last_burst=2
burst_range=1-2
if [[ -z "${NIGHTLY_GOLDEN_DIR:-}" ]]; then
    echo "NIGHTLY_GOLDEN_DIR is not set; generate and export the SNAP etad_ifg baseline first" >&2
    exit 1
fi

mkdir -p "$output_dir"
script_dir=$(cd -- "$(dirname -- "$0")" && pwd)

# Storm Daniel affected Thessaly in September 2023. Bursts 1-2 select stable, non-flooded scatterers in Thessaly.
# See STORM_DANIEL_ETAD.md for event references and SNAP preparation.
reference_slc="S1A_IW_SLC__1SDV_20230906T044011_20230906T044038_050202_060AD8_546A.SAFE"
reference_etad="S1A_IW_ETA__AXDV_20230906T044011_20230906T044038_050202_060AD8_7307.SAFE"
secondary_slc="S1A_IW_SLC__1SDV_20230918T044011_20230918T044038_050377_0610CC_620B.SAFE"
secondary_etad="S1A_IW_ETA__AXDV_20230918T044011_20230918T044038_050377_0610CC_2448.SAFE"

for product in "$reference_slc" "$secondary_slc"; do
    if [[ ! -d "$slc_files_dir/$product" ]]; then
        echo "Missing unpacked SLC product: $slc_files_dir/$product" >&2
        exit 1
    fi
done
for product in "$reference_etad" "$secondary_etad"; do
    if [[ ! -d "$etad_files_dir/$product" ]]; then
        echo "Missing unpacked ETAD product: $etad_files_dir/$product" >&2
        exit 1
    fi
done

reference_orbit="$orbit_files_dir/S1A_OPER_AUX_POEORB_OPOD_20230926T080700_V20230905T225942_20230907T005942.EOF"
secondary_orbit="$orbit_files_dir/S1A_OPER_AUX_POEORB_OPOD_20231008T080701_V20230917T225942_20230919T005942.EOF"
dem_files=(
    "$dem_files_dir/Copernicus_DSM_COG_10_N38_00_E021_00_DEM.tif"
    "$dem_files_dir/Copernicus_DSM_COG_10_N38_00_E022_00_DEM.tif"
    "$dem_files_dir/Copernicus_DSM_COG_10_N39_00_E021_00_DEM.tif"
    "$dem_files_dir/Copernicus_DSM_COG_10_N39_00_E022_00_DEM.tif"
)

for aux_file in "$reference_orbit" "$secondary_orbit" "${dem_files[@]}"; do
    if [[ ! -f "$aux_file" ]]; then
        echo "Missing prepared auxiliary file: $aux_file" >&2
        exit 1
    fi
done

golden_name="etad_ifg_IW1_VV_ref_b${burst_range}_sec_b${burst_range}_06Sep2023_18Sep2023.img"
golden="$NIGHTLY_GOLDEN_DIR/$golden_name"
golden_header="${golden%.img}.hdr"
if [[ ! -f "$golden" || ! -f "$golden_header" ]]; then
    echo "Missing SNAP pre-deburst etad_ifg golden: $golden" >&2
    echo "Generate it with SNAP as documented in build-automation/STORM_DANIEL_ETAD.md" >&2
    exit 1
fi

dem_args=()
for dem_file in "${dem_files[@]}"; do
    dem_args+=(--dem "$dem_file")
done

reference_stem=${reference_slc%.SAFE}
sidecar="$output_dir/${reference_stem}_IW1_VV_ref_b${burst_range}_sec_b${burst_range}_Orb_etad_Stack_ifg.tif"
difference="$output_dir/etad_ifg_IW1_VV_ref_b${burst_range}_sec_b${burst_range}_alus_minus_snap.tif"
mask_difference="$output_dir/etad_ifg_IW1_VV_ref_b${burst_range}_sec_b${burst_range}_mask_difference.tif"
rm -f "$sidecar" "$difference" "$mask_difference"

time alus-coh \
    -r "$slc_files_dir/$reference_slc" \
    -s "$slc_files_dir/$secondary_slc" \
    --orbit_ref "$reference_orbit" \
    --orbit_sec "$secondary_orbit" \
    "${dem_args[@]}" \
    --sw IW1 -p VV \
    --b_ref1 "$first_burst" --b_ref2 "$last_burst" \
    --b_sec1 "$first_burst" --b_sec2 "$last_burst" \
    --rg_win 10 --az_win 3 \
    --srp_number_points 501 --srp_polynomial_degree 5 --orbit_degree 3 \
    --subtract_flat_earth_phase true \
    --etad_ref "$etad_files_dir/$reference_etad" \
    --etad_sec "$etad_files_dir/$secondary_etad" \
    --wif --ll info \
    -o "$output_dir"

if [[ ! -f "$sidecar" ]]; then
    echo "Missing ALUs etad_ifg sidecar after successful processing: $sidecar" >&2
    exit 1
fi

# This first comparison reports measurements but has no acceptance thresholds. Add gates only after reviewing the
# independently generated SNAP baseline and retained difference rasters.
python3 "$script_dir/compare_etad_ifg.py" "$golden" "$sidecar" \
    --difference "$difference" \
    --mask-difference "$mask_difference" \
    --report-tolerance 0.001 \
    --report-tolerance 0.005 \
    --report-tolerance 0.01 \
    --report-tolerance 0.02 \
    --report-tolerance 0.05 \
    --report-tolerance 0.1
