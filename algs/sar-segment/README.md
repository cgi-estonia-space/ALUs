# Segmentation generation

It will combine all bands/polarisations (VV and VH) in order to generate 3 band false color (VV, VH, VH/VV) image.

Currently only Sentinel-1 GRD datasets are supported.

Processing steps below.

**Sentinel-1 GRD**
* Thermal noise removal
* Calibration
* Despeckle (optional)
* Range doppler terrain correction
* Division between VH/VV
* All 3 bands converted to dB

## Requirements

* ~20 GB RAM
* Min ~1.5 GB GPU memory

## Arguments

### alus-sar-segment

```
ALUs - SAR segment
Version 1.6.0

Arguments:

  -h [ --help ]          Print help
  -i [ --input ] arg     Input SAFE dataset (zipped or unpacked)
  -o [ --output ] arg    Output folder or filename
  -t [ --type ] arg      Type of calibration to be performed, one of the 
                         following - sigma;beta;gamma;dn
  --win arg              Despeckle window size for Refined Lee filter. Usually 
                         3,5...17. When unspecified no despeckle processed.
  --dem arg              DEM file(s). SRTM3 and Copernicus DEM 30m COG are 
                         currently supported.

  --ll arg (=verbose)    Log level, one of the following - 
                         verbose|debug|info|warning|error
  --gpu_mem arg (=100)   Percentage of how much GPU memory can be used for 
                         processing


https://github.com/cgi-estonia-space/ALUs


```

## Performance

It is strongly advised to use **GDAL_NUM_THREADS=ALL_CPUS** and **GDAL_CACHEMAX=12884901888** (~12GB) in order to enable
faster writing of the results. First it is large ~11GB, but with compression (LZW) applied it takes around ~8GB for full
landmass. Hence enabling GDAL driver to use all CPUs accelerates compressing and then using RAM for caching.

These options can be prepended to the call:

```
GDAL_NUM_THREADS=ALL_CPUS GDAL_CACHEMAX=12884901888 alus-sar-segment -i ....
```

Calculations itself take less than 20 seconds, it is heavily dependent on end results' writing speed.

## Results

Results highlight segmentation/contrasts of the SAR scene.

![alus-sar-segment-result-example1](https://alus-goods-set.s3.eu-central-1.amazonaws.com/alus_repo_docs/sar-segment/alus-sar-segment-result-example-1.png)