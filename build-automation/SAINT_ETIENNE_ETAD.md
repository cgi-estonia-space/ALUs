# Saint-Étienne ETAD comparison

This recipe is implemented by `run_saint_etienne_etad_test.sh` and `compare_etad_ifg.py`. The test exercises ETAD
preparation, fifth-plane backgeocoding, corrected coherence, `--wif` export and comparison with SNAP.

## Dataset and input reference

Use [ETAD Correction Tutorial v3](https://drive.google.com/file/d/1RYGGYMI-KvxDd_u40dITl-yp8sWQNjw9/view) ([backup link](https://github.com/user-attachments/files/33526763/ETAD.Correction.Tutorial.v3.pdf))
as the reference for ETAD datasets, matching SLC inputs and data preparation.

For this comparison, select the Saint-Etienne pair from 2024-08-12 (reference) and 2024-08-24 (secondary),
IW1 / VV / SLC burst 2 on both dates (ETAD global bIndex 5).

Input filenames, relative to the SLC and ETAD input directories:

```text
Reference SLC: S1A_IW_SLC__1SDV_20240812T173153_20240812T173220_055183_06B9C2_FAB2.SAFE
Reference ETA: S1A_IW_ETA__AXDV_20240812T173153_20240812T173220_055183_06B9C2_E046.SAFE
Secondary SLC: S1A_IW_SLC__1SDV_20240824T173153_20240824T173220_055358_06C040_6841.SAFE
Secondary ETA: S1A_IW_ETA__AXDV_20240824T173153_20240824T173220_055358_06C040_5B80.SAFE
```

The existing reader-stage SNAP outputs are:

```text
processed/S1A_IW_SLC__1SDV_20240812T173153_20240812T173220_055183_06B9C2_FAB2_split_Orb_etad.dim
processed/S1A_IW_SLC__1SDV_20240824T173153_20240824T173220_055358_06C040_6841_split_Orb_etad.dim
```

Each corresponding `.data/tie_point_grids/` directory contains `etadPhaseCorrection_IW1_5.img`,
`etadHeight_IW1_5.img` and `etadGradient_IW1_5.img`, with accompanying ENVI `.hdr` files.

Record exact orbit files, DEM inputs, SNAP module versions, graph parameters and checksums alongside the golden data.
Use the embedded processing graphs in the generated BEAM-DIMAP products to record the actual inputs and settings.
The E2E script accepts separate SLC and ETAD directories, prepared DEM and orbit directories, and an optional output
location. It reads the SNAP comparison layer from `NIGHTLY_GOLDEN_DIR`, matching the other local E2E runners.

## Generate SNAP goldens

For each acquisition:

1. Read the SLC `manifest.safe`.
2. TOPSAR-Split: subswath IW1, selected polarisation VV, firstBurstIndex=2, lastBurstIndex=2.
3. Apply-Orbit-File: precise orbit, polyDegree=3, continueOnFail=false. Preserve the actual orbit identifier.
4. S1-ETAD-Correction: supply that date's ETAD manifest, resamplingImage=false, outputPhaseCorrections=true.
5. Write BEAM-DIMAP. Keep the source acquisition timestamps in its name.

This produces the coarse `etadPhaseCorrection_IW1_5`, `etadHeight_IW1_5` and `etadGradient_IW1_5` TPGs used for the
reader investigation. The six supplied grids have already been examined; see
[precision results](../docs/etad.md#measured-native-precision-differences).

For the **full-resolution sidecar** golden:

1. Feed both ETAD-prepared products into Back-Geocoding, keeping the 12 August acquisition as reference.
2. Run the classical TOPS Interferogram path which produces `etad_ifg`. Save the pre-deburst radar-grid product in
   BEAM-DIMAP and export that named band, preserving its dimensions and validity mask.
3. Use the same orbital inputs and explicitly document DEM choices. In the inspected SNAP 14.x code, coarse ETAD TPG
   coregistration internally uses Copernicus 90 m; it is not necessarily controlled by the image backgeocoding DEM option.
4. Record any grid-generation differences between that SNAP path and ALUs' planned full-resolution secondary pixel map.

The quantity to compare is:

```text
reference_phase - coregistered_secondary_phase
    - coregistered_secondary_gradient * (reference_height - coregistered_secondary_height)
```

Do not compare this pair-specific, full-resolution plane directly against a single-acquisition 109×435 TPG, a debursted
plane, a terrain-corrected raster or the coherence magnitude. Those are different quantities or grids.

## ALUs run and comparison

The E2E script invokes `alus-coh` with `--etad_ref` and `--etad_sec`, matching IW1/VV/burst selection, explicit orbits
and DEMs, an output directory and `--wif`. It verifies processing success, then compares
`S1A_IW_SLC__1SDV_20240812T173153_20240812T173220_055183_06B9C2_FAB2_IW1_VV_ref_b2_sec_b2_Orb_etad_Stack_ifg.tif`
with the SNAP pre-deburst `etad_ifg` golden.

Use pixelwise comparison, following the evaluation approach documented in the original ALUs wiki and its `rastcomp`
comparisons. An adaptation of that raster-comparison workflow belongs in the E2E script, not another ETAD C++ integration
test executable. `alus_result_check.py` can check dimensions/metadata, but its strict golden comparison alone is not a
substitute for an ETAD-aware tolerance-based numerical comparison.

- Check dimensions, reference burst, pixel order, phase units and sign.
- Compare validity masks separately; zero phase is valid.
- Read both rasters into double-precision buffers without altering their stored values.
- Compute per-pixel error, maximum absolute error, RMSE, signed mean error, error quantiles and out-of-tolerance counts.
- Retain a difference raster and inspect spatial patterns; statistics alone can conceal alignment errors.
- Set tolerances from measured precision/interpolation effects with review, rather than requiring matching hashes.
- Preserve native ALUs precision. SNAP's intermediate Float32 input conversion is a documented difference, not a step
  to reproduce in ALUs or in golden generation.

The measured baseline has 33,740,991 mutually valid pixels, 70,509 SNAP-only valid pixels, 0 ALUs-only valid pixels,
0.0044522 rad RMSE, 0.017245 rad p99 absolute error and 0.17306 rad maximum absolute error. The committed gate allows
at most 71,000 SNAP-only valid pixels, zero ALUs-only valid pixels, 0.005 rad RMSE, 0.2 rad maximum error and 1,000
pixels over 0.1 rad. These tolerances are specific to the documented products, orbit files, DEMs and algorithms;
changing any input requires review rather than silently widening the limits. The comparator retains ALUs-minus-SNAP
and mask-mismatch rasters and reports error quantiles and counts at several thresholds.

The SNAP `etad_ifg` comparison validates correction sign and values. The SNAP `.img`/`.hdr` golden, prepared orbit files
and DEM tiles are checksum-pinned by the runner. `execute_e2e.sh` and `alus-golden/e2e-local.sh` register the case in the
dataset-enabled and local E2E suites respectively.
