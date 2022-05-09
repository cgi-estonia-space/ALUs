# coherence estimation

Coherence estimation routine means: 2 x input SLC coregistration(split + apply orbit file + backgeocoding) -> coherence estimation -> deburst -> range doppler terrain correction -> GTiff output  

## Arguments

#alus-coh

```
  -h [ --help ]                         Print help
  -r [ --in_ref ] arg                   Reference scene's input SAFE dataset 
                                        (zipped or unpacked)
  -s [ --in_sec ] arg                   Secondary scene's input SAFE dataset 
                                        (zipped or unpacked)
  --b_ref1 arg                          Reference scene's first burst index - 
                                        starting at '1', leave unspecified for 
                                        whole subswath
  --b_ref2 arg                          Reference scene's last burst index - 
                                        starting at '1', leave unspecified for 
                                        whole subswath
  --b_sec1 arg                          Secondary scene's first burst index - 
                                        starting at '1', leave unspecified for 
                                        whole subswath
  --b_sec2 arg                          Secondary scene's last burst index - 
                                        starting at '1', leave unspecified for 
                                        whole subswath
  --orbit_ref arg                       Reference scene's POEORB file
  --orbit_sec arg                       Secondary scenes's POEORB file
  -o [ --output ] arg                   Output folder or filename
  -p [ --polarisation ] arg             Polarisation for which coherence 
                                        estimation will be performed - VV;VH
  --sw arg                              Reference scene's subswath
  -a [ --aoi ] arg                      Area Of Interest WKT polygon, overrules
                                        first and last burst indexes
  --dem arg                             DEM file(s). Only SRTM3 is currently 
                                        supported.
  --no_mask_cor                         Do not mask out areas without elevation
                                        in coregistration
  --orbit_dir arg                       ESA SNAP compatible root folder of 
                                        orbit files. Can be used to find 
                                        correct one during processing. For 
                                        example: /home/<user>/.snap/auxData/Orb
                                        its/Sentinel-1/POEORB/
  --srp_number_points arg (=501)
  --srp_polynomial_degree arg (=5)
  --subtract_flat_earth_phase arg (=1)  Compute flat earth phase subtraction 
                                        during coherence operation. By default 
                                        on.
  --rg_win arg (=15)                    range window size in pixels.
  --az_win arg (=0)                     azimuth window size in pixels, if zero 
                                        derived from range window.
  --orbit_degree arg (=3)
  -w [ --wif ]                          Write intermediate results (will be 
                                        saved in the same folder as final 
                                        outcome). NOTE - this may decrease 
                                        performance. By default off.

  --ll arg (=verbose)                   Log level, one of the following - 
                                        verbose|debug|info|warning|error
  --gpu_mem arg (=100)                  Percentage of how much GPU memory can 
                                        be used for processing


```

#alus-coht

```
 -h [ --help ]                         Print help
  -i [ --input ] arg                    Timeline search directory
  -s [ --timeline_start ] arg           Timeline start - format YYYYMMDD
  -e [ --timeline_end ] arg             Timeline end - format YYYYMMDD
  -m [ --timeline_mission ] arg         Timeline mission filter - S1A or S1B
  -o [ --output ] arg                   Output folder or filename
  -p [ --polarisation ] arg             Polarisation for which coherence 
                                        estimation will be performed - VV;VH
  --sw arg                              Reference scene's subswath
  -a [ --aoi ] arg                      Area Of Interest WKT polygon, overrules
                                        first and last burst indexes
  --dem arg                             DEM file(s). Only SRTM3 is currently 
                                        supported.
  --no_mask_cor                         Do not mask out areas without elevation
                                        in coregistration
  --orbit_dir arg                       ESA SNAP compatible root folder of 
                                        orbit files. Can be used to find 
                                        correct one during processing. For 
                                        example: /home/<user>/.snap/auxData/Orb
                                        its/Sentinel-1/POEORB/
  --srp_number_points arg (=501)
  --srp_polynomial_degree arg (=5)
  --subtract_flat_earth_phase arg (=1)  Compute flat earth phase subtraction 
                                        during coherence operation. By default 
                                        on.
  --rg_win arg (=15)                    range window size in pixels.
  --az_win arg (=0)                     azimuth window size in pixels, if zero 
                                        derived from range window.
  --orbit_degree arg (=3)
  -w [ --wif ]                          Write intermediate results (will be 
                                        saved in the same folder as final 
                                        outcome). NOTE - this may decrease 
                                        performance. By default off.

  --ll arg (=verbose)                   Log level, one of the following - 
                                        verbose|debug|info|warning|error
  --gpu_mem arg (=100)                  Percentage of how much GPU memory can 
                                        be used for processing
```





