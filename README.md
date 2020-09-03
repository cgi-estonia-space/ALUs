# alus

The go to software for SAR Image GPU processing.

# Build

```
cmake -H. -Bbuild
cd build
make -j8
```

# Running

Main executable that loads and runs algorithms from shared libraries (.so) is located at
'build/alus_package'.

Coherence example:
```
./aluserer --alg_name coherence -i ~/coherence_datasets/4_bands.tif -o /tmp/coherence_test.tif --aux ~/coherence_datasets
```

Command line arguments:
```
Alus options:
  -h [ --help ]                   Print help
  --alg_name arg                  Specify algorithm to run
  -i [ --input ] arg              Input dataset path/name. GeoTIFF files only.
  -o [ --output ] arg             Output dataset path/name
  -x [ --tile_width ] arg (=500)  Tile width.
  -y [ --tile_height ] arg (=500) Tile height.
  --aux arg                       Auxiliary file locations (metadata, incident 
                                  angle, etc).
  -c [ --conf ] arg               Algorithm specific configuration. Must be 
                                  supplied as key=value pairs separated by 
                                  whitespace.
  -l [ --list_algs ]              Print available algorithms
```

# Testing

```
cd build/test
./unit_test
```
