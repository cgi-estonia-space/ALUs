# ETAD reader: numerical precision and SNAP comparison

## Implementation policy

The ETAD reader preserves the source samples as `double` and performs phase, interpolation and gradient calculations
in double precision. It does **not** narrow inputs to Float32 to reproduce SNAP's implementation. Matching algorithms,
units, grids and validity handling is the objective; bitwise reproduction of an intermediate precision loss is not.

`sentinel1_etad_product.cc` handles native XML and grouped NetCDF input. The owning grids and metadata are plain C++
structures, not SNAP `Product` or `MetadataElement` objects. `etad_grid.h` provides non-owning grid views and geometry;
`etad_computation.cuh` contains inline host/device scalar calculations without GDAL, XML, filesystem or Ceres dependencies.
Ceres is used only for SAFE/ZIP access and extraction lifetime.

## Context from the original ALUs work

The original `cgi-estonia-space/ALUs` wiki already treats small floating-point differences as a numerical evaluation
problem rather than an automatic algorithm failure:

- [Coherence estimation evaluation — Single subswath](https://github.com/cgi-estonia-space/ALUs/wiki/Coherence-estimation-evaluation#single-subswath)
  reports that almost all valid pixels can differ in a decimal place while the average relative difference is around
  0.1%. It discusses orbital interpolation, backgeocoding and coherence arithmetic as sources of differences. The page
  links the original [GPU-GSTP-MPR-0008 report](https://github.com/user-attachments/files/17943084/GPU-GSTP-MPR-0008.pdf).
- [Calibration routine evaluation — Single subswath](https://github.com/cgi-estonia-space/ALUs/wiki/Calibration-routine-evaluation#single-subswath)
  reports pixelwise differing percentages and relative error alongside aggregate statistics. Its SRTM3 example has
  1.615699% differing pixels and 2.689596 PPM average relative difference, with zero pixels in its "bad pixels" category.
- [Document stash](https://github.com/cgi-estonia-space/ALUs/wiki/Document-stash) indexes the earlier reports and figures.

These pages were consulted on 2026-09-11. Their older coherence/calibration figures are context, **not ETAD acceptance
thresholds**. In particular, GPU/CPU execution differences should not be used as a blanket explanation: the native ETAD
comparison below used host calculations, and a specific SNAP reader conversion was identified.

## Inputs and exact comparison stage

For ETAD datasets, matching SLC inputs and data preparation, refer to
[ETAD Correction Tutorial v3](https://drive.google.com/file/d/1RYGGYMI-KvxDd_u40dITl-yp8sWQNjw9/view).

The comparison below uses these Saint-Etienne acquisitions:

| Role | SLC SAFE filename | ETAD SAFE filename |
|---|---|---|
| Reference, 2024-08-12 | `S1A_IW_SLC__1SDV_20240812T173153_20240812T173220_055183_06B9C2_FAB2.SAFE` | `S1A_IW_ETA__AXDV_20240812T173153_20240812T173220_055183_06B9C2_E046.SAFE` |
| Secondary, 2024-08-24 | `S1A_IW_SLC__1SDV_20240824T173153_20240824T173220_055358_06C040_6841.SAFE` | `S1A_IW_ETA__AXDV_20240824T173153_20240824T173220_055358_06C040_5B80.SAFE` |

Relative to the test dataset directory, the inspected SNAP products are `processed/<SLC stem>_split_Orb_etad.dim`
and the corresponding `.data` directories. Here `<SLC stem>` is the SLC filename without `.SAFE`.
Their embedded processing graphs specify:

- TOPSAR-Split: IW1, VV, first and last SLC burst both 2.
- Apply-Orbit-File: Sentinel Precise, polynomial degree 3.
- S1-ETAD-Correction: `resamplingImage=false`, `outputPhaseCorrections=true`.
- SLC radar frequency: `5405.000454334349 MHz`, explicitly converted to Hz for calculation.

The selected SLC burst 2 corresponds to **ETAD global bIndex 5**, not bIndex 2.

For each date, the three compared arrays were:

```text
processed/<SLC stem>_split_Orb_etad.data/tie_point_grids/etadPhaseCorrection_IW1_5.img
processed/<SLC stem>_split_Orb_etad.data/tie_point_grids/etadHeight_IW1_5.img
processed/<SLC stem>_split_Orb_etad.data/tie_point_grids/etadGradient_IW1_5.img
```

Each is 109 azimuth rows × 435 range columns = **47,415 samples**, stored by SNAP as Float32 ENVI grids. These are
single-acquisition, pre-backgeocoding grids. No coherence or interferogram values enter this comparison.

## Measured native-precision differences

ALUs reads the four native NetCDF arrays, then computes:

```text
phase = -2*pi*f*(troposphere + geodetic_range - ionosphere + range_calibration)
height = ETAD height
gradient = -2*pi*f*regression_slope(range_differences(troposphere), range_differences(height))
```

The first table compares ALUs double results directly with the numerical values stored in SNAP's Float32 grids.

| Date | Layer | Units | Maximum absolute difference, ALUs double vs SNAP |
|---|---|---|---:|
| 2024-08-12 | phase | rad | 6.345925635287131e-5 |
| 2024-08-12 | height | m | 6.103374471422285e-5 |
| 2024-08-12 | gradient | rad/m | 1.775183393601387e-6 |
| 2024-08-24 | phase | rad | 6.313106314337347e-5 |
| 2024-08-24 | height | m | 6.102655038375815e-5 |
| 2024-08-24 | gradient | rad/m | 1.233243256515415e-6 |

For a second, storage-only diagnostic, the already-computed ALUs result was rendered to Float32. This conversion
was **after** computation, not inside the reader or calculation. It separates output storage effects from input
quantisation effects. All six arrays have 47,415 finite samples and zero invalid-mask mismatches.

| Date | Layer | Differing samples after Float32 rendering | Max absolute difference | RMSE |
|---|---|---:|---:|---:|
| 2024-08-12 | phase | 11,857 | 6.103515625e-5 | 3.052176141431886e-5 |
| 2024-08-12 | height | 0 | 0 | 0 |
| 2024-08-12 | gradient | 43,350 | 1.773238182067871e-6 | 1.028258083687429e-7 |
| 2024-08-24 | phase | 11,716 | 6.103515625e-5 | 3.032517070862533e-5 |
| 2024-08-24 | height | 0 | 0 | 0 |
| 2024-08-24 | gradient | 43,279 | 1.236796379089355e-6 | 7.528342643150978e-8 |

### Causes

The inspected Microwave Toolbox source is `origin/14.x` at
`c2637806de2477ca0af2e23e3d669679b20d9f4d`:

1. `sar-io/.../Sentinel1ETADNetCDFReader.java`, `readDataForRank2Variable`, lines 299–325, reads through
   `srcArray.getFloat(...)` and `destBuffer.setElemFloatAt(...)`, even for Float64 NetCDF variables. Promoting these
   values back to double later cannot recover discarded bits.
2. `sar-op-sentinel1/.../etadcorrectors/TOPSCorrector.java`, `computeRangeTimeCorrectionPhase`, performs the phase formula
   on these narrowed values. Multiplication by approximately `2*pi*5.405e9` amplifies small delay differences into
   observable phase differences.
3. `ETADUtils.java`, `computeGradientForCurrentBurst` / `computeGradient`, uses differences between neighbouring samples
   and a local regression with an intercept. Rounding of both height and troposphere affects those differences. A high
   fraction of non-identical gradient samples is therefore compatible with a small absolute discrepancy.
4. `TOPSCorrector.saveBurstDataAsTiePointGrid` converts the computed values to Float32 again for TPG storage. Height's
   exact match after output rendering is consistent with storage rounding alone.

A temporary diagnostic experiment rounded the four ALUs inputs before calculation. After output rendering it produced
zero differing values for all six grids. This strongly isolates the reader's Float32 conversion as the source of the
observed differences for these data. **That experiment was removed from the implementation.** It is evidence about the
cause, not a production precision policy.

The results support the indexing, orientation, sign and regression implementation for these two bursts. They do not
establish universal tolerances or validate the later full-resolution sidecar, resampling or coherence path.

## Calibration schema clarification

The earlier variable named `legacy` was a `pugi::xml_node`, not a version number or calibration value. Pugixml converts
an existing node to true and a missing node to false. Consequently:

```cpp
legacy ? node : reference
```

selected the per-channel record for the old direct-value schema, or the product reference element for the newer schema.
The implementation now uses `direct_calibrations`, `has_direct_calibrations` and explicit branches.

- Direct-value schema: `instrumentTimingCalibrationList/instrumentTimingCalibration` carries `rangeCalibration` and
  `azimuthCalibration` for each swath/polarisation.
- Reference-plus-offset schema: `instrumentTimingCalibrationReference` carries the common values, and
  `instrumentTimingCalibrationOffsetList` carries channel-dependent `rangeOffset` and `azimuthOffset` values.
- The parser now **retains both reference values and offsets separately**. The current InSAR preparation uses the
  reference range calibration. It does not apply channel offsets.

The old comment about using only the reference "for SNAP 14.x parity" described the inspected SNAP utility's behavior;
it was not evidence that nonzero offsets can generally be discarded. The supplied `support/etadProduct.xsd` documents
channel-dependent offsets (lines 1322–1336 and 2384–2421) and separate per-burst residual polarisation offsets to apply
relative to the reference channel (lines 341–390). A future polarisation-aware correction must distinguish these records
and account for what SETAP has already incorporated into the grids, rather than blindly adding both.

The inspected Saint-Etienne 2024-08-12 annotation has zero channel offsets and declares no swath/polarisation-dependent
timing information. These comparisons therefore do not resolve nonzero-offset application. That remains an explicit
extension beyond the current reference-calibration phase preparation.

## Geolocation scope

The parser retains geographic corners, temporal coverage, sampling, grid extents and timing-calibration metadata.
`LoadLayer` can read a named two-dimensional array such as `lats`, `lons` or a geometric correction layer. However,
`LoadInSarBurstLayers` prepares only phase, height and troposphere-to-height gradient. It does not calculate corrected
pixel positions, combine the geometric correction layers or resample complex imagery.

## Verification and future E2E

Committed unit tests cover only annotation parsing and input-metadata combinations. The temporary numerical/CUDA test
cases and the `etad_grid_compare.cc` executable have been removed as requested. Numerical acceptance will be through the
coherence E2E with explicit SLC/ETAD/orbit/DEM inputs and an exported `--wif` sidecar.

The comparison **method** is reusable: read corresponding raster windows, promote values to double, compare masks and
each pixel, and report maximum absolute error, RMSE, signed bias, quantiles and counts above justified tolerances.
The old executable's DIMAP-specific discovery of burst TPGs is not the final sidecar reader.

Aggregate statistics are not guaranteed to match across data types: quantisation affects means, extrema, variance and
subsequent calculations. Conversely, equal statistics can conceal a transposed or shifted raster. Pixelwise comparison
on the same grid is essential. Reading a Float32 golden into a double buffer is fine; narrowing ALUs inputs is not.

See [ETAD E2E data recipe](../build-automation/ETAD_E2E.md) for the staged SNAP graph and future sidecar comparison.

Reader-step verification commands, with `BUILD_DIR` set to the configured build directory:

```bash
cmake --build "$BUILD_DIR" --target sentinel1-etad-unit-test -j12
"$BUILD_DIR/unit-test/sentinel1-etad-unit-test"
ctest --test-dir "$BUILD_DIR" -R '^sentinel1-etad-unit-test$' --output-on-failure
cmake --build "$BUILD_DIR" -j12
```

The six parser/input-structure cases passed, the focused CTest entry passed, and the full build succeeded. Formatting
and `git diff --check` passed. Clang-tidy was also run during development; it reported style warnings, including
recommendations conflicting with this repository's explicit `#pragma once` convention. No ETAD coherence E2E was run:
the pipeline and `--wif` sidecar integration are a later step.

## Artifacts from this investigation

The native-precision calculations were rendered to Float32 GeoTIFFs for the second table. The separate input-rounding
experiment produced diagnostic artifacts only; those do not represent the final implementation.

SHA-256 of the native-calculation rendered artifacts, identified independently of storage location:

| Acquisition date | Artifact | SHA-256 |
|---|---|---|
| 2024-08-12 | `etadGradient_IW1_5.tif` | `3efbc7c68235f9e6cd049ed96a7e43f93a11cc82a6499c4b211a7ebd2b4e1063` |
| 2024-08-12 | `etadHeight_IW1_5.tif` | `4c8017775f8f2d1ff4110c2584c2f407d23bb4a9595ecedc7256b20cf3394c2d` |
| 2024-08-12 | `etadPhaseCorrection_IW1_5.tif` | `df633b4685d14020a2df02884543f570ba1761625ebdeddcd24d3e7408671f4a` |
| 2024-08-24 | `etadGradient_IW1_5.tif` | `eec525ab99c279768144284c00af22d976e32caaae53f59d824b6cb820aa577b` |
| 2024-08-24 | `etadHeight_IW1_5.tif` | `34c68b00d2a88092e869934c5f00ce7dbd6cd10e6ec6bfff464b9041c9a778cc` |
| 2024-08-24 | `etadPhaseCorrection_IW1_5.tif` | `0e707ae29d0ce2d2993abd35facb7734befef574c9bb8fae3e79d44ffb76330d` |
