# ALUS

Collection of processing operators/routines defined by software - plural of ALU(Arithmetic Logic Unit)  
or  
ALU for Space/Surveillance etc.

A software project that targets to utilize Nvidia GPUs for processing earth observation data (faster).  
Kickstart of this project was funded
through [ESA's EOEP programme](http://www.esa.int/About_Us/Business_with_ESA/Business_Opportunities/Earth_Observation_Envelope_Programme)  
Current development is funded
through [ESA's GSTP programme](https://www.esa.int/Enabling_Support/Space_Engineering_Technology/Shaping_the_Future/About_the_General_Support_Technology_Programme_GSTP)

Developed by [CGI Estonia](https://www.cgi.com/ee/et).

## [Quick performance overview](PERFORMANCE.md)
For further comprehensive evaluation see [Wiki](https://github.com/cgi-estonia-space/ALUs/wiki) 

# Out of the box usage

Verified releases can be downloaded from - https://github.com/cgi-estonia-space/ALUs/releases/
One can download docker image with all of the needed dependencies
from [dockerhub](https://hub.docker.com/repository/docker/cgialus/alus-devel) or simply `docker pull cgialus/alus-devel`

## Executing

Each algorithm is a separate executable. Currently available ones are (more info and usage in parenthesis):

* Sentinel 1 coherence estimation routine - ``alus-coh`` ([README](algs/coherence-estimation-routine/README.md))
* Sentinel 1 coherence estimation timeline generation - ``alus-coht`` ([README](algs/coherence-estimation-routine/README.md))
* Sentinel 1 calibration routine - ``alus-cal`` ([README](algs/calibration-routine/README.md))
* Sentinel 2 and other raster resample and tiling - ``alus-resa`` ([README](algs/resample/README.md))
* Gabor feature extraction - ``alus-gfe`` ([README](algs/feature-extraction-gabor/README.md))
* PALSAR level 0 focuser - ``alus-palsar-focus`` ([README](algs/palsar-focus/README.md))
* SAR segmentation generation - ``alus-sar-segment`` ([README](algs/sar-segment/README.md))

When building separately, these are located at ``<build_dir>/alus_package``

Update **PATH** environment variable in order to execute everywhere:  
``export PATH=$PATH:/path/to/<alus_package>``

See ``--help`` for specific arguments/parameters how to invoke processing. For more information see detailed explanation
of Sentinel 1 processors' [processing arguments](docs/PROCESSING_ARGUMENTS.md).

## Docker example

NVIDIA driver and NVIDIA Container Toolkit must be installed together with docker.

```
docker pull cgialus/alus-devel:latest
docker run -t -d --gpus all --name alus_container cgialus/alus-devel
docker exec -t alus_container mkdir /root/alus
docker cp <latest build tar archive> alus_container:/root/alus/
docker exec -t alus_container bash -c "tar -xzf /root/alus/*.tar.gz -C /root/alus/"
# Use docker cp to transfer all the input datasets, auxiliary data, then either
docker exec -t alus_container bash -c "cd /root/alus; ./alus-<alg> ...."
# Or connect to shell
docker exec -it alus_container /bin/bash
# Running coherence estimation routine example
./alus-coh -r S1A_IW_SLC__1SDV_20200724T034334_20200724T034401_033591_03E49D_96AA.SAFE \
-s S1A_IW_SLC__1SDV_20200805T034334_20200805T034401_033766_03E9F9_52F6.SAFE \
-o /tmp/ -p VV --orbit_dir <orbit files location> --sw IW1 --dem srtm_43_06.tif --dem srtm_44_06.tif
# Running calibration routine example
./alus-cal -i S1A_IW_SLC__1SDV_20180815T154813_20180815T154840_023259_028747_4563.SAFE \
-o /tmp/alus_S1A_IW_SLC__1SDV_20180815T154813_20180815T154840_023259_028747_4563_Calib_tc.tif \
--sw IW1 -p VV --type beta --dem srtm_42_01.tif
```

# Dependencies

[Setup of dependencies](DEPENDENCIES.md)

# Building

```
cmake . -Bbuild
cd build
make -j8
```

# Jupyter Notebook

There is a Jupyter Notebook located at `jupyter-notebook` folder with a user-friendly interface and automated auxiliary file downloads.
It can be used in conjunction with binaries to easily execute code. Read the [instructions](jupyter-notebook/README.md).

# Minimum/Recommended requirements

For specific, check each processor's README.

Below are rough figures:
* NVIDIA GPU device compute capability 6.0 (Pascal) or higher
* 2(minimum)/4(recommended) GB of available device memory (some ALUs can manage with less)
* High speed (NVMe) SSD to benefit from the computation speedups
* 4 GB of extra RAM to enable better caching/input-output (GDAL raster IO)

# Contributing

[Contribution guidelines](CONTRIBUTING.md)

# License

[GNU GPLv3](LICENSE.txt)

# [Release notes](RELEASE.md)

[Binary downloads](https://github.com/cgi-estonia-space/ALUs/releases/) 
