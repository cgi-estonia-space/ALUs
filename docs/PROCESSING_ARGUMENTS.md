
# Processing arguments

## Processed results

Some algorithms provide ```--wif``` parameter to save intermediate results, by default only a final result is saved.
Hence specifying ``--wif`` will enable getting intermediate results in the same folder as final result, whereas default behavior
will not produce any extra files.

Final output can be specified with ``-o`` argument where directory value does not influence the output name. By default an output name will be
input plus processing steps acronym. One can also define final result filename, then the value should be fully qualified filename, for example: 
``-o /tmp/S1A_IW_SLC__1SDV_20200904T111450_20200904T111520_034208_03F970_E017_coh_window2010.tif``

## Coherence estimation (``alus-coh``)

``--help`` output

```
  -h [ --help ]                     Print help
  -r [ --in_ref ] arg               Reference scene's input SAFE dataset 
                                    (zipped or unpacked)
  -s [ --in_sec ] arg               Secondary scene's input SAFE dataset 
                                    (zipped or unpacked)
  -o [ --output ] arg               Output folder or filename
  -p [ --polarisation ] arg         Polarisation for which coherence estimation
                                    will be performed - VV;VH
  --sw arg                          Reference scene's subswath
  --b_ref1 arg                      Reference scene's first burst index - 
                                    starting at '1', leave unspecified for 
                                    whole subswath
  --b_ref2 arg                      Reference scene's last burst index - 
                                    starting at '1', leave unspecified for 
                                    whole subswath
  --b_sec1 arg                      Secondary scene's first burst index - 
                                    starting at '1', leave unspecified for 
                                    whole subswath
  --b_sec2 arg                      Secondary scene's last burst index - 
                                    starting at '1', leave unspecified for 
                                    whole subswath
  -a [ --aoi ] arg                  Area Of Interest WKT polygon, overrules 
                                    first and last burst indexes
  --dem arg                         DEM file(s). Only SRTM3 is currently 
                                    supported.
  --orbit_ref arg                   Reference scene's POEORB file
  --orbit_sec arg                   Secondary scenes's POEORB file
  --orbit_dir arg                   ESA SNAP compatible root folder of orbit 
                                    files. Can be used to find correct one 
                                    during processing. For example: 
                                    /home/<user>/.snap/auxData/Orbits/Sentinel-
                                    1/POEORB/
  --srp_number_points arg (=501)
  --srp_polynomial_degree arg (=5)
  --subtract_flat_earth_phase       Compute flat earth phase subtraction during
                                    coherence operation. By default on.
  --rg_win arg (=15)                range window size in pixels.
  --az_win arg (=0)                 azimuth window size in pixels, if zero 
                                    derived from range window.
  --orbit_degree arg (=3)
  -w [ --wif ]                      Write intermediate results (will be saved 
                                    in the same folder as final outcome). NOTE 
                                    - this may decrease performance. By default
                                    off.

  --ll arg (=verbose)               Log level, one of the following - 
                                    verbose|debug|info|warning|error
  --gpu_mem arg (=100)              Percentage of how much GPU memory can be 
                                    used for processing
```

For orbit files there are multiple options. If a user has a SNAP installation and has processed the same scenes already, the existing orbit files
can be used by supplying SNAP's orbit file directory e.g. ``--orbit_dir  /home/<user>/.snap/auxData/Orbits/Sentinel-1/POEORB/``.
If such option does not exist or there are no specific orbit files present, one can use [sentineleof](https://github.com/scottstanie/sentineleof) and then supply downloaded files via ``--orbit_ref`` and ``--orbit_sec``.

When specifying an area to be processed traditional subswath and burst index parameters can be used. However, it might be simpler
to use ``-a``/``--aoi`` parameter to specify exact region. This must be a WKT polygon and does not have to follow burst boundaries exactly.
For example a stripe like polygon can be supplied where it will consider all the bursts to be processed which are overlapping with the given coordinates.
Currently a subswath parameter must be supplied (this requirement will be removed in upcoming versions) along with area of interest.
For analyzing scenes' layout and position on map a [S-1 TOPS SPLIT Analyzer](https://github.com/pbrotoisworo/s1-tops-split-analyzer) could be used.

Coherence computation parameters are pure computation specific and further knowledge of these could be found in SNAP Help/Documentation or via some other resources in web.
This document here does not discuss specifics of those - since these are algorithm specific math details.

### Examples

**Full subswath (IW1) with orbit files from SNAP aux.**
```
alus-coh -r S1A_IW_SLC__1SDV_20200724T034334_20200724T034401_033591_03E49D_96AA.SAFE \
-s S1A_IW_SLC__1SDV_20200805T034334_20200805T034401_033766_03E9F9_52F6.SAFE \
--orbit_dir ~/.snap/auxData/Orbits/Sentinel-1/POEORB/ --sw IW1 -p VV \
--dem srtm_43_06.tif --dem srtm_44_06.tif -o /tmp/ --ll info
```

**AOI based (3 bursts covered) processing with orbit files and specifying final result filename. Inputs as ZIP files.**
```
alus-coh -r S1B_IW_SLC__1SDV_20210615T054959_20210615T055026_027363_0344A0_83FE.zip \
-s S1A_IW_SLC__1SDV_20210703T055050_20210703T055117_038609_048E45_35F7.zip \
--orbit_ref S1B_OPER_AUX_POEORB_OPOD_20210705T111814_V20210614T225942_20210616T005942.EOF.zip \
--orbit_sec S1A_OPER_AUX_POEORB_OPOD_20210723T121923_V20210702T225942_20210704T005942.EOF.zip \
--sw IW3 -p VV --ll info --dem srtm_37_02.zip --dem srtm_37_03.zip --dem srtm_38_02.zip --dem srtm_38_03.zip \
--aoi "POLYGON ((3.76064043932478 50.6679002753201,4.81930157970497 50.5884971985178,4.65806260842462 50.0309601054367,3.65031903792243 50.1622939049033,3.76064043932478 50.6679002753201))" \
-o /tmp/S1B_IW_SLC__1SDV_20210615T054959_20210615T055026_027363_0344A0_83FE_Orb_Stack_coh_deb_tc_zipped.tif 
```

## Calibration (``alus-cal``)

``--help`` output

```
  -h [ --help ]              Print help
  -i [ --input ] arg         Input SAFE dataset (zipped or unpacked)
  -o [ --output ] arg        Output folder or filename
  -w [ --wif ]               Write intermediate results (will be saved in the 
                             same folder as final outcome). NOTE - this may 
                             decrease performance. Default OFF.
  -s [ --subswath ] arg      Subswath for which the calibration will be 
                             performed, one of the following - IW1;IW2;IW3
  -p [ --polarisation ] arg  Polarisation for which the calibration will be 
                             performed - VV;VH
  --bi1 arg                  First burst index - starting at '1', leave 
                             unspecified for whole subswath
  --bi2 arg                  Last burst index - starting at '1', leave 
                             unspecified for whole subswath
  -a [ --aoi ] arg           Area Of Interest WKT polygon, overrules first and 
                             last burst indexes
  -t [ --type ] arg          Type of calibration to be performed, one of the 
                             following - sigma;beta;gamma;dn
  --dem arg                  DEM file(s). Only SRTM3 is currently supported.

  --ll arg (=verbose)        Log level, one of the following - 
                             verbose|debug|info|warning|error
  --gpu_mem arg (=100)       Percentage of how much GPU memory can be used for 
                             processing
```

Input of the processing is provided by ``-i``/``--input`` argument. The area on which to perform processing can be traditional subswath and burst index parameters.
However, it might be simpler to use ``-a``/``--aoi`` parameter to specify exact region. This must be a WKT polygon and does not have to follow burst boundaries exactly.
For example a stripe like polygon can be supplied where it will consider all the bursts to be processed which are overlapping with the given coordinates.
Currently a subswath parameter must be supplied (this requirement will be removed in upcoming versions) along with area of interest.

``-t``/``--type`` option specify which kind of calibration is performed - which look-up table is used and normalization formula - to radiometrically calibrate the data.

### Examples

**Gamma calibration of 4 bursts, ZIP SAFE input and specified output name**
```
alus-cal -i $test_dataset_dir/S1A_IW_SLC__1SDV_20210722T005537_20210722T005604_038883_049695_2E58.SAFE \
-o /tmp/S1A_IW_SLC__1SDV_20180815T154813_20180815T154840_023259_028747_4563_Calib_IW1_tc.tif  \
--sw IW2 --polarisation VV -t gamma --bi1 2 --bi2 6 \
--dem srtm_51_09.tif --dem srtm_52_09.tif --ll info
```
