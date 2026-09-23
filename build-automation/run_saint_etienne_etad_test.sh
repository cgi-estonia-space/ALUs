#!/bin/bash

set -euo pipefail

function print_help {
    echo "Usage:"
    echo "$0 <Saint-Etienne dataset dir> <COPDEM 30m COG dir> <orbit files dir> [output dir]"
}

if [[ $# -lt 3 || $# -gt 4 ]]; then
    echo "Wrong count of input arguments" >&2
    print_help
    exit 1
fi

dataset_dir=$1
dem_files_dir=$2
orbit_files_dir=$3
if [[ $# -eq 4 ]]; then
    output_dir=$4
else
    script_name=${0##*/}
    output_dir="/tmp/${script_name%.*}"
fi

mkdir -p "$output_dir"
script_dir=$(cd -- "$(dirname -- "$0")" && pwd)

# SNAP golden recipe: TOPSAR-Split IW1/VV burst 2, Apply-Orbit-File, S1-ETAD-Correction in grids-only mode,
# Back-Geocoding with Copernicus 30 m, then Interferogram. Compare the named etad_ifg band before debursting.
# ETAD's global bIndex is 5 for this selection; the ALUs CLI receives the one-based SLC burst index 2.

reference_slc="S1A_IW_SLC__1SDV_20240812T173153_20240812T173220_055183_06B9C2_FAB2.SAFE"
reference_etad="S1A_IW_ETA__AXDV_20240812T173153_20240812T173220_055183_06B9C2_E046.SAFE"
secondary_slc="S1A_IW_SLC__1SDV_20240824T173153_20240824T173220_055358_06C040_6841.SAFE"
secondary_etad="S1A_IW_ETA__AXDV_20240824T173153_20240824T173220_055358_06C040_5B80.SAFE"

function verify_checksum {
    local path=$1
    local expected=$2
    local actual
    actual=$(sha256sum "$path")
    actual=${actual%% *}
    if [[ "$actual" != "$expected" ]]; then
        echo "Checksum mismatch for $path: expected $expected, got $actual" >&2
        exit 1
    fi
}

for product in "$reference_slc" "$reference_etad" "$secondary_slc" "$secondary_etad"; do
    if [[ ! -d "$dataset_dir/$product" ]]; then
        echo "Missing unpacked SAFE product: $dataset_dir/$product" >&2
        exit 1
    fi
done

golden="$dataset_dir/processed/${reference_slc%.SAFE}_split_Orb_etad_Stack_ifg.data/etad_ifg_IW1_VV_12Aug2024_24Aug2024.img"
golden_header="${golden%.img}.hdr"
if [[ ! -f "$golden" || ! -f "$golden_header" ]]; then
    echo "Missing SNAP pre-deburst etad_ifg golden: $golden" >&2
    exit 1
fi
verify_checksum "$golden" "71f7a52b7d999c27bb34421d6c5e2770885b538a029a60c9169257e1c1eb8fde"
verify_checksum "$golden_header" "cab2fc124822625f75e99ff401a885013bb415632ec36893802bc027a7de7e3c"

dem_basenames=(
    "Copernicus_DSM_COG_10_N45_00_E003_00_DEM.tif"
    "Copernicus_DSM_COG_10_N45_00_E004_00_DEM.tif"
)
dem_args=()
mkdir -p "$output_dir/dem"
for basename in "${dem_basenames[@]}"; do
    source_dem="$dem_files_dir/$basename"
    if [[ ! -f "$source_dem" ]]; then
        echo "Missing DEM file: $source_dem" >&2
        exit 1
    fi
    # ALUs currently tokenizes DEM argument values on spaces, so use stable local links.
    ln -sfn "$source_dem" "$output_dir/dem/$basename"
    dem_args+=(--dem "$output_dir/dem/$basename")
done
verify_checksum "$dem_files_dir/${dem_basenames[0]}" \
    "6f0d215f4df43ad728481a1ebac50183e1800413abd15f776938c756c97d288f"
verify_checksum "$dem_files_dir/${dem_basenames[1]}" \
    "ccebea0a93d3ba6bfa3da64e1c309d7b80ba306ee8d1808ce9cbba7bd8b4c27f"

function resolve_orbit {
    local basename=$1
    local eof_checksum=$2
    local zip_checksum=$3
    if [[ -f "$orbit_files_dir/$basename" ]]; then
        verify_checksum "$orbit_files_dir/$basename" "$eof_checksum"
        printf '%s\n' "$orbit_files_dir/$basename"
        return
    fi
    if [[ -f "$orbit_files_dir/$basename.zip" ]]; then
        verify_checksum "$orbit_files_dir/$basename.zip" "$zip_checksum"
        unzip -p "$orbit_files_dir/$basename.zip" "$basename" > "$output_dir/$basename"
        verify_checksum "$output_dir/$basename" "$eof_checksum"
        printf '%s\n' "$output_dir/$basename"
        return
    fi
    echo "Missing orbit file: $orbit_files_dir/$basename or $orbit_files_dir/$basename.zip" >&2
    exit 1
}

reference_orbit=$(resolve_orbit \
    "S1A_OPER_AUX_POEORB_OPOD_20240901T070629_V20240811T225942_20240813T005942.EOF" \
    "5f362d28503ebd1b111bdd108ea8f0180c6467bac955d2cb2419ab52dc393bc7" \
    "8dfadb02976a11c153ad5acf0b9b6d395b3cbae4f0c7f8ee9b5efae919e9e6b0")
secondary_orbit=$(resolve_orbit \
    "S1A_OPER_AUX_POEORB_OPOD_20240913T070636_V20240823T225942_20240825T005942.EOF" \
    "0e24c422adb500094ee1fc434bafe6957bbe758b18cbc0fc40e6197d3da9a05b" \
    "183cc23e1d14ca361a43042f9ca873551f6cfdb4225b3156b6a3b7a9903edcac")
output="$output_dir/saint-etienne-etad-coherence.tif"
sidecar="$output_dir/saint-etienne-etad-coherence_Orb_Stack_etad_ifg.tif"
corrected_coherence="$output_dir/saint-etienne-etad-coherence_Orb_Stack_coh.tif"
uncorrected_output="$output_dir/saint-etienne-no-etad-coherence.tif"
uncorrected_coherence="$output_dir/saint-etienne-no-etad-coherence_Orb_Stack_coh.tif"

rm -f "$output" "$sidecar" "$corrected_coherence" "$uncorrected_output" "$uncorrected_coherence" \
    "$output_dir/etad_ifg_alus_minus_snap.tif" "$output_dir/etad_ifg_mask_difference.tif"

common_args=(
    -r "$dataset_dir/$reference_slc"
    -s "$dataset_dir/$secondary_slc"
    --orbit_ref "$reference_orbit"
    --orbit_sec "$secondary_orbit"
    "${dem_args[@]}"
    --sw IW1 -p VV
    --b_ref1 2 --b_ref2 2
    --b_sec1 2 --b_sec2 2
    --rg_win 10 --az_win 3
    --srp_number_points 501 --srp_polynomial_degree 5 --orbit_degree 3
    --subtract_flat_earth_phase true
    --wif --ll info
)

time alus-coh "${common_args[@]}" \
    --etad_ref "$dataset_dir/$reference_etad" \
    --etad_sec "$dataset_dir/$secondary_etad" \
    -o "$output"

if [[ ! -f "$sidecar" ]]; then
    echo "Missing ALUs etad_ifg sidecar after successful processing: $sidecar" >&2
    exit 1
fi

python3 "$script_dir/compare_etad_ifg.py" "$golden" "$sidecar" \
    --difference "$output_dir/etad_ifg_alus_minus_snap.tif" \
    --mask-difference "$output_dir/etad_ifg_mask_difference.tif" \
    --absolute-tolerance 0.1 \
    --max-outliers 1000 \
    --rmse-tolerance 0.005 \
    --max-absolute-error 0.2 \
    --max-mask-mismatches 71000 \
    --max-snap-only-valid 71000 \
    --max-alus-only-valid 0 \
    --report-tolerance 0.001 \
    --report-tolerance 0.005 \
    --report-tolerance 0.01 \
    --report-tolerance 0.02 \
    --report-tolerance 0.05

# A second real-scene run guards the phase plane's connection to coherence, not just sidecar generation.
time alus-coh "${common_args[@]}" -o "$uncorrected_output"
python3 "$script_dir/check_etad_coherence_effect.py" "$corrected_coherence" "$uncorrected_coherence" \
    --sample-x 6478 --sample-y 736 \
    --expected-corrected 0.32724872 --sample-tolerance 0.001 \
    --minimum-max-change 0.05
