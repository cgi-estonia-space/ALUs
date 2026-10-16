# Storm Daniel ETAD comparison

`run_storm_daniel_etad_test.sh` processes a Sentinel-1 pair over Thessaly, Greece, and compares the ALUs
full-resolution `etad_ifg` sidecar with an independently generated SNAP baseline. SNAP processing is a prerequisite;
the runner does not create or substitute the baseline.

## Event and scene selection

[Daniel](https://en.wikipedia.org/wiki/Storm_Daniel) - the name of the storm. The 6 September 2023 acquisition occurred during the destructive rainfall
and flooding in Thessaly. The 18 September acquisition is the same-relative-orbit, 12-day post-event comparison.

- [Copernicus Emergency Mapping activation EMSR692](https://mapping.emergency.copernicus.eu/activations/EMSR692/)
  documents flooding in Greece beginning on 5 September and provides mapped areas around the Thessaly floodplain.
- [NASA Earth Observatory: A Deluge in Greece](https://science.nasa.gov/earth/earth-observatory/a-deluge-in-greece-151807/)
  describes the rainfall and reports Sentinel-1-observed flooding in Thessaly on 6 September.
- [Dynamics, predictability, impacts and climate change considerations of the catastrophic Mediterranean Storm Daniel
  (2023)](https://doi.org/10.5194/wcd-6-1515-2025) provides a meteorological and impact assessment.
- [Storm Daniel fueled by anomalously high sea surface temperatures in the Mediterranean](https://doi.org/10.1038/s41612-024-00872-2)
  analyses the event's atmospheric and oceanic conditions.

The validation target is stable, non-flooded scatterers in Thessaly, not the flood extent itself. The selection is
IW1/VV/SLC bursts 1–2, corresponding to ETAD global `bIndex` values 2 and 5. It covers the southern Thessaly scene
containing Karditsa and adjoining western mountain terrain. Urban and industrial surfaces, infrastructure and exposed
rock outside mapped inundation provide the useful controls.

The [EMSR692 activation data](https://rapidmapping.emergency.copernicus.eu/backend/dashboard-api/public-activations/?code=EMSR692)
show that the Palamas and Karditsa flood AOIs intersect bursts 1–2. Those AOIs are used to identify areas to exclude or
report separately. Floodwater, wet agricultural soil, vegetation changes and damage can independently reduce coherence,
so a scene-wide coherence change must not be attributed solely to ETAD.

## Inputs

| Role | SLC | ETAD |
|---|---|---|
| Event/reference, 2023-09-06 | `S1A_IW_SLC__1SDV_20230906T044011_20230906T044038_050202_060AD8_546A.SAFE` | `S1A_IW_ETA__AXDV_20230906T044011_20230906T044038_050202_060AD8_7307.SAFE` |
| Post-event/secondary, 2023-09-18 | `S1A_IW_SLC__1SDV_20230918T044011_20230918T044038_050377_0610CC_620B.SAFE` | `S1A_IW_ETA__AXDV_20230918T044011_20230918T044038_050377_0610CC_2448.SAFE` |

Prepared auxiliaries are:

```text
S1A_OPER_AUX_POEORB_OPOD_20230926T080700_V20230905T225942_20230907T005942.EOF
S1A_OPER_AUX_POEORB_OPOD_20231008T080701_V20230917T225942_20230919T005942.EOF
Copernicus_DSM_COG_10_N38_00_E021_00_DEM.tif
Copernicus_DSM_COG_10_N38_00_E022_00_DEM.tif
Copernicus_DSM_COG_10_N39_00_E021_00_DEM.tif
Copernicus_DSM_COG_10_N39_00_E022_00_DEM.tif
```

SAFE products are used unpacked; orbit files and DEM tiles are used directly.

## Generate the SNAP baseline first

Follow the operator configuration in ESA's
[ETAD Correction Tutorial v3](https://drive.google.com/file/d/1RYGGYMI-KvxDd_u40dITl-yp8sWQNjw9/view).
Use the same burst range for both acquisitions:

1. Read each SLC `manifest.safe`.
2. Run TOPSAR-Split with subswath IW1, polarisation VV, `firstBurstIndex=1` and `lastBurstIndex=2`.
3. Run Apply-Orbit-File with the matching precise orbit, polynomial degree 3 and `continueOnFail=false`.
4. Run S1-ETAD-Correction with that acquisition's ETAD product, `resamplingImage=false` and
   `outputPhaseCorrections=true`.
5. Write both prepared acquisitions as BEAM-DIMAP, retaining their processing graphs.
6. Back-geocode the pair with 6 September as reference and 18 September as secondary. Use the same Copernicus DEM
   source and orbit inputs recorded above.
7. Run the classical Interferogram operator to produce `etad_ifg`. Preserve the pre-deburst radar-grid product; do not
   compare a debursted, terrain-corrected or coherence raster with the ALUs sidecar.
8. Export only the `etad_ifg` band as ENVI, preserving its dimensions and validity mask.

Name the exported pair as follows:

```text
etad_ifg_IW1_VV_ref_b1-2_sec_b1-2_06Sep2023_18Sep2023.img
etad_ifg_IW1_VV_ref_b1-2_sec_b1-2_06Sep2023_18Sep2023.hdr
```

Place the `.img` and `.hdr` together in the directory passed through `NIGHTLY_GOLDEN_DIR`. Record the SNAP version,
module versions, graph, input filenames and output checksums with the baseline.

## Run ALUs and compare

Run from the repository root after placing `alus-coh` on `PATH`:

```bash
PATH="$BUILD_DIR/alus_package:$PATH" \
NIGHTLY_GOLDEN_DIR=<SNAP golden dir> \
build-automation/run_storm_daniel_etad_test.sh \
    <SLC dir> <ETAD dir> <COPDEM 30m dir> <orbit dir> [output dir]
```

The script checks all prepared inputs, runs `alus-coh` with `--wif`, and compares the named ALUs sidecar with the SNAP
baseline using `compare_etad_ifg.py`. It retains ALUs-minus-SNAP and validity-mask mismatch rasters. The initial run only
reports pixelwise metrics; acceptance thresholds and checksum pins must be added after the independent SNAP products and
difference rasters have been reviewed.

SNAP's classical coherence branch does not apply `etad_ifg`, so it is not an ETAD-corrected coherence baseline. A later
coherence experiment must compare ETAD and non-ETAD ALUs runs on the same coregistration path. Stable urban,
infrastructure and exposed-rock masks are the primary validation sets; floodplain, wet soil and vegetation masks are
excluded or reported separately.
