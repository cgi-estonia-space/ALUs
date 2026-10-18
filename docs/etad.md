# ETAD InSAR integration: numerical precision and SNAP comparison

## Implementation policy

The ETAD reader preserves the source samples as `double` and performs phase, interpolation and gradient calculations
in double precision. It does **not** reproduce the Float32 conversion in SNAP's
[rank-2 NetCDF reader](https://github.com/senbox-org/microwave-toolbox/blob/c2637806de2477ca0af2e23e3d669679b20d9f4d/sar-io/src/main/java/eu/esa/sar/io/sentinel1/Sentinel1ETADNetCDFReader.java#L299-L324).
Matching algorithms, units, grids and validity handling is the objective; bitwise reproduction of an intermediate
precision loss is not.

`sentinel1_etad_product.cc` handles native XML and grouped NetCDF input. The owning grids and metadata are plain C++
structures, not SNAP `Product` or `MetadataElement` objects. `etad_grid.h` provides non-owning grid views and geometry;
`etad_computation.cuh` contains inline host/device scalar calculations without GDAL, XML, filesystem or Ceres dependencies.
Ceres is used only for SAFE/ZIP access and extraction lifetime.

## SLC association and preparation

Host-side SLC association and ETAD preparation are at `sentinel1/etad`:
- `slc_metadata.h` and `slc_metadata.cc` read one native SLC product annotation selected by swath and polarisation.
  Plain C++ metadata is returned without SNAP constructs like `Product`, `MetadataElement`, `Sentinel1Utils` or
  `SubSwathInfo`. Radar frequency remains in Hz and native `slantRangeTime` remains a two-way time in seconds.
- `etad_preparation.h` and `etad_preparation.cc` associate a zero-based contiguous SLC burst selection with global ETAD
  burst indices, validate complete coverage and load the existing double-precision phase, height and gradient layers.

SLC-to-ETAD association does not call the generic point lookup `FindBurst`: adjacent ETAD burst coverages overlap.
Instead it follows [SNAP TOPS preparation](https://github.com/senbox-org/microwave-toolbox/blob/c2637806de2477ca0af2e23e3d669679b20d9f4d/sar-op-sentinel1/src/main/java/eu/esa/sar/sentinel1/gpf/etadcorrectors/TOPSCorrector.java#L281-L289)
by matching an SLC burst first-line time to the ETAD burst azimuth minimum within 0.1 seconds. The
[ALUs association](https://github.com/kptr-juku/ALUs/blob/6e94a565bd4ffe54646dd86208435695e0d00f0f/sentinel1/etad/etad_preparation.cc#L88-L116)
additionally requires exactly one match and verifies that the complete SLC burst azimuth and two-way range extents lie
strictly inside that ETAD burst's coverage.

Standard SAFE names select the ETAD acquisition by their sensing-time and orbit token. Renamed or burst-product inputs
fall back to the native annotation mission, absolute orbit and start/stop interval, requiring one unambiguous ETAD
acquisition. The fallback allows one second around ETAD input-product bounds because those timestamps can be rounded to
whole seconds while native SLC annotation bounds retain microseconds.

The preparation entry point is independent of TOPSAR-Split and the SNAP data model:

```cpp
const auto slc_metadata = slc::ReadMetadata(slc_path, "IW1", "VV");
const auto etad_product = Sentinel1EtadProduct::Open(etad_path);
// first_burst is zero-based, so {1, 1} selects source burst 2.
const auto prepared = etad::PrepareInSar(slc_metadata, {1, 1}, etad_product);
```

Each `PreparedBurst` owns its correction arrays and records the source SLC index, split-local index, output line offset,
global ETAD `bIndex`, SLC timing geometry and ETAD interpolation geometry. Reference and secondary acquisitions are
prepared independently; pair correspondence and device transfer are done by backgeocoding operator.

Timing-calibration lookup requires both swath and polarisation. It selects the matching direct/reference value but
does not add the separately retained channel offset; nonzero channel offsets are not currently supported.

## Inputs and exact comparison stage

The input products and processing configurations are from ESA's official
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

`bIndex` is product-global rather than a local SLC burst number. In this product, each temporal group is ordered IW3,
IW1, IW2: the first group has indices 1, 2, 3 and the second has 4, 5, 6. The selected IW1 SLC burst 2 therefore
corresponds to **ETAD global bIndex 5**. Association uses burst time rather than assuming this ordering.

For each date, the three compared arrays are:

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

For a second, storage-only diagnostic, the already-computed ALUs result is in Float32. This conversion
is done **after** computation, not inside the reader or calculation. It separates output storage effects from input
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

1. [`Sentinel1ETADNetCDFReader.readDataForRank2Variable`](https://github.com/senbox-org/microwave-toolbox/blob/c2637806de2477ca0af2e23e3d669679b20d9f4d/sar-io/src/main/java/eu/esa/sar/io/sentinel1/Sentinel1ETADNetCDFReader.java#L299-L324) reads through
   `srcArray.getFloat(...)` and `destBuffer.setElemFloatAt(...)`, even for Float64 NetCDF variables. Promoting these
   values back to double later cannot recover discarded bits.
2. [`TOPSCorrector.computeRangeTimeCorrectionPhase`](https://github.com/senbox-org/microwave-toolbox/blob/c2637806de2477ca0af2e23e3d669679b20d9f4d/sar-op-sentinel1/src/main/java/eu/esa/sar/sentinel1/gpf/etadcorrectors/TOPSCorrector.java#L317-L334)
   performs the phase formula on these narrowed values. Multiplication by approximately `2*pi*5.405e9` amplifies small
   delay differences into observable phase differences.
3. [`ETADUtils.computeGradientForCurrentBurst` and `computeGradient`](https://github.com/senbox-org/microwave-toolbox/blob/c2637806de2477ca0af2e23e3d669679b20d9f4d/sar-op-sentinel1/src/main/java/eu/esa/sar/sentinel1/gpf/etadcorrectors/ETADUtils.java#L544-L587)
   use differences between neighbouring samples and a local regression with an intercept. Rounding of both height and
   troposphere affects those differences. A high fraction of non-identical gradient samples is therefore compatible
   with a small absolute discrepancy.
4. [`TOPSCorrector.saveBurstDataAsTiePointGrid`](https://github.com/senbox-org/microwave-toolbox/blob/c2637806de2477ca0af2e23e3d669679b20d9f4d/sar-op-sentinel1/src/main/java/eu/esa/sar/sentinel1/gpf/etadcorrectors/TOPSCorrector.java#L304-L314)
   converts the computed values to Float32 again for TPG storage. Height's exact match after output rendering is
   consistent with storage rounding alone.

A temporary diagnostic experiment rounded the four ALUs inputs before calculation. After output rendering it produced
zero differing values for all six grids. This strongly isolates the reader's Float32 conversion as the source of the
observed differences for these data. The diagnostic is not part of the production implementation and does not define
its precision policy.

For these two bursts, the results support the indexing, orientation, sign and regression implementation. Universal
tolerances and the later full-resolution sidecar, resampling and coherence path require separate validation.

## Geolocation scope

The parser retains geographic corners, temporal coverage, sampling, grid extents and timing-calibration metadata.
`LoadLayer` can read a named two-dimensional array such as `lats`, `lons` or a geometric correction layer. However,
`LoadInSarBurstLayers` prepares only phase, height and troposphere-to-height gradient. It does not calculate corrected
pixel positions, combine the geometric correction layers or resample complex imagery.

## Pipeline integration and verification

Unit tests cover only annotation parsing and input-metadata combinations. Pairwise `alus-coh` accepts
`--etad_ref` and `--etad_sec`, which must be supplied together. Selected acquisitions are prepared independently and
the correction grids for both acquisitions are uploaded to GPU memory during backgeocoding. ETAD runs add a fifth
Float32 plane after reference I/Q and secondary I/Q:

```text
etad_ifg = reference_phase - secondary_phase
    - secondary_gradient * (reference_height - secondary_height)
```

Secondary quantities are sampled with the backgeocoding pixel map. Coherence rotates each valid secondary complex
sample by `exp(+j * etad_ifg)` before flat-earth correction and accumulation. Non-finite corrections zero the secondary
sample and are represented as NaN in the fifth plane; zero phase remains valid. Non-ETAD and timeline runs retain the
four-plane path. With `--wif`, the pre-deburst fifth plane is exported with description `etad_ifg`, radians units, NaN
no-data and ETAD provenance.

## E2E comparison method

Use ESA's [ETAD Correction Tutorial v3](https://drive.google.com/file/d/1RYGGYMI-KvxDd_u40dITl-yp8sWQNjw9/view)
and the same inputs in SNAP and ALUs. Prepare each acquisition in SNAP with TOPSAR-Split, Apply-Orbit-File and
S1-ETAD-Correction (`resamplingImage=false`, `outputPhaseCorrections=true`). Back-geocode the pair, run the classical
Interferogram operator and preserve the pre-deburst radar-grid product. Export only `etad_ifg` as ENVI. Do not
substitute a single-acquisition TPG, debursted plane, terrain-corrected raster or coherence magnitude.

Record the SNAP and module versions, processing graph, input filenames, orbit files, DEMs and checksums with the golden.
Compare dimensions and pixel order first, then finite-value masks and mutually valid pixels. Promote stored samples to
double and report maximum absolute error, RMSE, signed bias, quantiles and out-of-tolerance counts. Zero phase is valid.
Retain ALUs-minus-SNAP and mask-mismatch rasters. Tolerances are fixture-specific and must not be widened without
review.

Both runners require unpacked SAFE products and prepared auxiliary files. Put `alus-coh` on `PATH`, set
`NIGHTLY_GOLDEN_DIR` to the SNAP baseline directory and invoke them from the repository root:

```bash
PATH="$BUILD_DIR/alus_package:$PATH" NIGHTLY_GOLDEN_DIR=<SNAP golden dir> \
build-automation/run_saint_etienne_etad_test.sh \
    <SLC dir> <ETAD dir> <COPDEM 30m dir> <orbit dir> [output dir]

PATH="$BUILD_DIR/alus_package:$PATH" NIGHTLY_GOLDEN_DIR=<SNAP golden dir> \
build-automation/run_storm_daniel_etad_test.sh \
    <SLC dir> <ETAD dir> <COPDEM 30m dir> <orbit dir> [output dir]
```

### Saint-Étienne accepted fixture

The products and IW1/VV/burst-2 selection are listed above. Prepared auxiliaries are:

```text
S1A_OPER_AUX_POEORB_OPOD_20240901T070629_V20240811T225942_20240813T005942.EOF
S1A_OPER_AUX_POEORB_OPOD_20240913T070636_V20240823T225942_20240825T005942.EOF
Copernicus_DSM_COG_10_N45_00_E003_00_DEM.tif
Copernicus_DSM_COG_10_N45_00_E004_00_DEM.tif
```

The SNAP golden is `etad_ifg_IW1_VV_12Aug2024_24Aug2024.{img,hdr}`. The runner checksum-pins the golden and auxiliaries.
The accepted baseline has 33,740,991 mutually valid pixels, 70,509 SNAP-only valid pixels, no ALUs-only valid pixels,
0.0044522 rad RMSE, 0.017245 rad p99 absolute error and 0.17306 rad maximum absolute error. Gates are 71,000
SNAP-only valid pixels, no ALUs-only valid pixels, 0.005 rad RMSE, 0.2 rad maximum error and at most 1,000 pixels over
0.1 rad.

### Storm Daniel exploratory fixture

Use IW1/VV bursts 1–2, with 6 September 2023 as reference and 18 September as secondary:

```text
Reference SLC:  S1A_IW_SLC__1SDV_20230906T044011_20230906T044038_050202_060AD8_546A.SAFE
Reference ETAD: S1A_IW_ETA__AXDV_20230906T044011_20230906T044038_050202_060AD8_7307.SAFE
Secondary SLC:  S1A_IW_SLC__1SDV_20230918T044011_20230918T044038_050377_0610CC_620B.SAFE
Secondary ETAD: S1A_IW_ETA__AXDV_20230918T044011_20230918T044038_050377_0610CC_2448.SAFE
```

Prepared auxiliaries are the matching precise orbits
`S1A_OPER_AUX_POEORB_OPOD_20230926T080700_V20230905T225942_20230907T005942.EOF` and
`S1A_OPER_AUX_POEORB_OPOD_20231008T080701_V20230917T225942_20230919T005942.EOF`, plus Copernicus tiles N38E021,
N38E022, N39E021 and N39E022. Export the SNAP baseline as
`etad_ifg_IW1_VV_ref_b1-2_sec_b1-2_06Sep2023_18Sep2023.{img,hdr}`. This fixture reports metrics without acceptance
gates until its independent baseline, difference rasters and checksum pins have been reviewed.

The reference acquisition coincides with the Thessaly flooding documented by
[Copernicus EMSR692](https://mapping.emergency.copernicus.eu/activations/EMSR692/). Coherence validation must use stable,
non-flooded urban and industrial surfaces, infrastructure and exposed rock. Floodwater, wet agricultural soil,
vegetation changes and damaged areas must be excluded or reported separately.

## Coherence interpretation

ALUs applies `secondary *= exp(+j * etad_ifg)` before flat-earth correction and coherence-window accumulation. A
constant phase does not change coherence magnitude; only phase variation inside the 10 x 3 window changes the coherent
sum. SNAP's classical `InterferogramOp` applies ETAD in its
[interferogram branch](https://github.com/senbox-org/microwave-toolbox/blob/ff8c42a86935f4129f495be4481d5d93df9bb683/sar-op-insar/src/main/java/eu/esa/sar/insar/gpf/InterferogramOp.java#L5018-L5042),
but its [coherence branch](https://github.com/senbox-org/microwave-toolbox/blob/ff8c42a86935f4129f495be4481d5d93df9bb683/sar-op-insar/src/main/java/eu/esa/sar/insar/gpf/InterferogramOp.java#L5044-L5086)
reloads the secondary tile and applies only flat-earth and optional topographic phase. The same separation is present in
[SNAP 14.x](https://github.com/senbox-org/microwave-toolbox/blob/c2637806de2477ca0af2e23e3d669679b20d9f4d/sar-op-insar/src/main/java/eu/esa/sar/insar/gpf/InterferogramOp.java#L3207-L3273).

The tutorial-derived SNAP ETAD and non-ETAD DIMAP graphs also produced different coregistered secondary samples.
Consequently, the coherence difference measures graph/coregistration provenance, not an ETAD coherence effect. On
the matched ALUs stack, the smaller coherence change is the actual effect of the smooth ETAD phase inside each window.
The sign of a raster-wide mean change is not a phase-sign test.

The SNAP operator sequences and parameters are recorded in the generated DIMAP graphs. Both use IW1/VV, burst 2,
10 x 3 coherence and `Interferogram` with `includeCoherence=true`; the ETAD graph additionally includes
`S1-ETAD-Correction` in grids-only mode before `Back-Geocoding`.

## Reproduction

A direct Saint-Étienne ALUs run is:

```bash
./alus-coh \
    -r S1A_IW_SLC__1SDV_20240812T173153_20240812T173220_055183_06B9C2_FAB2.SAFE/manifest.safe \
    -s S1A_IW_SLC__1SDV_20240824T173153_20240824T173220_055358_06C040_6841.SAFE/manifest.safe \
    --b_ref1 2 --b_ref2 2 --b_sec1 2 --b_sec2 2 --rg_win 10 --az_win 3 --sw IW1 -p VV \
    --orbit_ref S1A_OPER_AUX_POEORB_OPOD_20240901T070629_V20240811T225942_20240813T005942.EOF \
    --orbit_sec S1A_OPER_AUX_POEORB_OPOD_20240913T070636_V20240823T225942_20240825T005942.EOF \
    --dem Copernicus_DSM_COG_10_N45_00_E003_00_DEM.tif \
    --dem Copernicus_DSM_COG_10_N45_00_E004_00_DEM.tif \
    --etad_ref S1A_IW_ETA__AXDV_20240812T173153_20240812T173220_055183_06B9C2_E046.SAFE \
    --etad_sec S1A_IW_ETA__AXDV_20240824T173153_20240824T173220_055358_06C040_5B80.SAFE \
    -o processed/ --wif
```

Verification commands, with `BUILD_DIR` set to the configured build directory, are:

```bash
cmake --build "$BUILD_DIR" --target sentinel1-etad-unit-test -j12
"$BUILD_DIR/unit-test/sentinel1-etad-unit-test"
ctest --test-dir "$BUILD_DIR" -R '^sentinel1-etad-unit-test$' --output-on-failure
cmake --build "$BUILD_DIR" -j12
ctest --test-dir "$BUILD_DIR" --output-on-failure
```

The eleven parser/input-structure and association cases pass, all CTest targets pass, and the full build succeeds. The
Saint-Étienne E2E result and gates are recorded above.
