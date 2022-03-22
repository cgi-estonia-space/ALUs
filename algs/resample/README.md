# Resample

In order to up- and down-sample rasters, this is the processor to use. It leverages [NVIDIA's Performance Primitives
library](https://developer.nvidia.com/npp) for resampling operations. Tiling, reprojection and output format
specification is supported. In general any additional I/O operation (like tiling) would mean loss in end-to-end
performance. It supports Sentinel-2 datasets and any other rasters. Currently there is limitation for input and output
data types, supported ones are - signed/unsigned 16-bit integers, 8-bit integers and 32-bit floats.

## Arguments

```
--help                    Print help
-i [ --input ] arg        Input dataset(s)
-d [ --destination ] arg  Results output path
--dim_band arg            Dimension of the resampled image(s) taken from the
                          specified band's dimensions of the (first) input.
                          Alternative would be to manually specify using
                          '--dim' or '--width' and '-height' arguments.
--dim arg                 Dimension of the resampled image(s)
                          '{width}x{height}' in pixels. For example -
                          '1000x1500'. Alternatively can use '--width' and
                          '--height' arguments or '--dim_band'.
----width arg             Width of the resampled image(s). Must be specified
                          together with '--height' argument. Alternatively
                          can use '--dim' or '--dim_band' arguments.
----height arg            Width of the resampled image(s). Must be specified
                          together with '--width' argument. Alternatively can
                          use '--dim' or '--dim_band' arguments.
--tile_dim arg            Output tiles' dimension '{width}x{height}' in
                          pixels. For example - '100x50'. Alternatively can
                          use '--tile_width' and '--tile_height' arguments.
--tile_width arg          Output tiles' width. Alternatively can use
                          '--tile_dim'.
--tile_height arg         Output tiles' height. Alternatively can use
                          '--tile_dim'.
--overlap arg (=0)        Tiles overlap in pixels
-m [ --method ] arg       Resampling method, one of the following -
                          cubic, cubic2p-bspline, cubic2p-c05c03,
                          cubic2p-catmullrom, lanczos, lanczos3, linear,
                          nearest-neighbour, smooth-edge, super
-f [ --format ] arg       One of the following - GTiff, netCDF. Or any other
                          supported GDAL driver. Leave empty to use input
                          format.
-p [ --crs ] arg          Coordinate reference system/projection for output
                          tiles. Leave empty to use input CRS. Consult
                          GDAL/PROJ for supported ones. In general this value
                          is supplied to SetFromUserInput() GDAL API. Some
                          valid examples - 'WGS84', 'EPSG:4326'.
--exclude arg             Bands to be excluded from resampling. When multiple
                          inputs are specified which differ in band count a
                          warning is given if band number specified is not
                          present. Starting from 1. Argument can be specified
                          multiple times.

--ll arg (=verbose)       Log level, one of the following -
                          verbose|debug|info|warning|error
--gpu_mem arg (=100)      Percentage of how much GPU memory can be used for
                          processing
```

## Performance

Reference laptop computer details:  
CPU: Intel i7 10750h\
RAM: 32GB\
GPU: NVIDIA GeForce GTX 1660 Ti 6GB\
SSD (NVMe): SAMSUNG MZALQ512HALU-000L2 

Compared against GDAL 3.0.4 respective functionality/tools on Ubuntu 20.04

### Resampling only

Two different resampling methods calculation wise (fast and slow one) were tested - no tiling.
Tiling would add I/O time and this would mean more similar results performance wise, while not exclusively testing
computation performance.

#### Nearest-neighbour

```
gdalwarp -r near -ts 10980 10980 T35VNE_20211102T093049_B8A.jp2 tiled/T35VNE_20211102T093049_B8A.tif
```
~3 seconds

```
./alus-resa -i S2B_MSIL1C_20211102T093049_N0301_R136_T35VNE_20211102T114211.SAFE -d /tmp/alus-resa --dim_band 2 
--tile_dim 10980x10980 -m nearest-neighbour --exclude 1 --exclude 2 --exclude 3 --exclude 4 --exclude 5 --exclude 6 
--exclude 7 --exclude 8 --exclude 10 --exclude 11 --exclude 12 --exclude 13 -f Gtiff
```
~1.5-2.5 seconds

##### Using RAM disk
GDAL - ~2.0 seconds\
ALUs - ~1.0 second


#### Lanczos 
```
gdalwarp -r lanczos -ts 10980 10980 T35VNE_20211102T093049_B8A.jp2 tiled/T35VNE_20211102T093049_B8A.tif
```
~17 seconds
```
./alus-resa -i S2B_MSIL1C_20211102T093049_N0301_R136_T35VNE_20211102T114211.SAFE 
-d /tmp/alus-resa --dim_band 2 --tile_dim 10980x10980 -m lanczos --exclude 1 --exclude 2 --exclude 3 --exclude 4 
--exclude 5 --exclude 6 --exclude 7 --exclude 8 --exclude 10 --exclude 11 --exclude 12 --exclude 13 -f Gtiff
```
~2-3 seconds

##### Using RAM disk
GDAL - ~17 seconds\
ALUs - ~1.1 seconds



