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

# Out of the box usage

Latest build can be downloaded from - https://alus-builds.s3.eu-central-1.amazonaws.com/alus-nightly-latest.tar.gz  
Verified releases can be downloaded from - https://bitbucket.org/cgi-ee-space/alus/downloads/  
One can download prepared image with all of the needed dependencies
from [dockerhub](https://hub.docker.com/repository/docker/cgialus/alus-devel)

## Executing

Each algorithm is a separate executable. Currently available ones are:

* Sentinel 1 coherence estimation routine - ``alus-coh``
* Sentinel 1 calibration routine - ``alus-cal``
* Gabor feature extraction - ``alus-gfe``

They are located at ``<build_dir>/alus_package``

Update **PATH** environment variable in order to execute everywhere:  
``export PATH=$PATH:/path/to/<alus_package>``

See ``--help`` for specific arguments/parameters how to invoke processing. For more information see detailed explanation
of [processing arguments](docs/PROCESSING_ARGUMENTS.md).

## Docker example

```
docker pull cgialus/alus-devel:latest
docker run -t -d --gpus all --name alus_container cgialus/alus-devel
docker exec -t alus_container mkdir /root/alus
docker cp <latest build tar archive> alus_container:/root/alus/
docker exec -t alus_container bash -c "tar -xzf /root/alus/*.tar.gz -C /root/alus/"
# Use docker cp to transfer all the input datasets, auxiliary data.
# docker exec -t alus_container bash -c "cd /root/alus; ./alus-<alg> ...."
# or
# docker exec -it alus_container /bin/bash to enter shell
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

There is a Jupyter Notebook with a user-friendly interface included into the repository.
It can be used in conjunction with binaries to easily execute code.

## Requirements

- [Core requirements](DEPENDENCIES.md)
- Python 3.8+ for executing this notebook
- python3-tk package

## Running instructions

It is best to run the notebook from its own python virtual environment. Environment can be created with
command  `python3 -m venv env` and it can be activated by the following command `source env/bin/activate`. In order to
deactivate the virtual environment use `deactivate`.

- Install all the necessary requirements:

`pip install -r requirements.txt`

- Launch Jupyter Notebook or Jupyter Lab:

`jupyter notebook` or `jupyter lab`

- Execute all the cells in order providing all the necessary input.

# [Performance](PERFORMANCE.md)

# Contributing

[Contribution guidelines](CONTRIBUTING.md)

# License

[GNU GPLv3](LICENSE.txt)

# [Release notes](RELEASE.md)

[Binary downloads](https://bitbucket.org/cgi-ee-space/alus/downloads/?tab=downloads)  
