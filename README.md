# ALUS

Acceleration Library Used for Sentinel.

A software project that targets to utilize Nvidia GPUs for processing earth observation data (faster).

Kickstart of this project was funded through ESA's EOEP programme - http://www.esa.int/About_Us/Business_with_ESA/Business_Opportunities/Earth_Observation_Envelope_Programme

Developed by CGI Estonia.

# Out of the box usage

Latest build can be downloaded from - https://alus-builds.s3.eu-central-1.amazonaws.com/alus-nightly-latest.tar.gz

On dockerhub one can download prepared image with all of the needed dependencies from [dockerhub](https://hub.docker.com/repository/docker/cgialus/alus-infra) 


## Docker example

```
docker pull cgialus/alus-infra:latest
docker run -t -d --gpus all --name alus_container cgialus/alus-infra
docker exec -t alus_container mkdir /root/alus
docker cp <latest build tar archive> alus_container:/root/alus/
docker exec -t alus_container bash -c "tar -xzf /root/alus/*.tar.gz -C /root/alus/"
# Use docker cp to transfer all the input datasets, auxiliary data.
# docker exec -t alus_container bash -c "cd /root/alus; ./alus --alg_name ...."
# or
# docker exec -it alus_container /bin/bash to enter shell
# Running coherence estimation routine example
./alus --alg_name coherence-estimation-routine -i S1A_IW_SLC__1SDV_20200724T034334_20200724T034401_033591_03E49D_96AA.SAFE -i S1A_IW_SLC__1SDV_20200805T034334_20200805T034401_033766_03E9F9_52F6.SAFE -o /tmp/ -p main_scene_identifier=S1A_IW_SLC__1SDV_20200724T034334,orbit_file_dir=<orbit files location>,subswath=IW1,polarization=VV --dem srtm_43_06.tif --dem srtm_44_06.tif --tile_width 2000 --tile_height 3000
# Running calibration routine example
./alus --alg_name calibration-routine -i S1A_IW_SLC__1SDV_20180815T154813_20180815T154840_023259_028747_4563.SAFE -o /tmp/alus_S1A_IW_SLC__1SDV_20180815T154813_20180815T154840_023259_028747_4563_Calib_tc.tif -x 5000 -y 5000 -p "subswath=IW1,polarisation=VV,calibration_type=beta" --dem srtm_42_01.tif
```

# Dependencies

[Setup of dependencies](DEPENDENCIES.md)

# Building

```
cmake . -Bbuild
cd build
make -j8
```

# Executing

Main executable with algorithm shared libraries(.so) is located at **<build_dir>/alus_package**.

In order to load shared library components one should move to directory where binaries are located or

``LD_LIBRARY_PATH=$LD_LIBRARY_PATH:...<build_dir>/alus_package; export LD_LIBRARY_PATH``

or move/create symlinks to a location that is present in default ``LD_LIBRARY_PATH``.

See ``--help`` and ``--alg_help`` for specific arguments/parameters how to invoke processing.

# [Performance](PERFORMANCE.md)

# Contributing

[Contribution guidelines](CONTRIBUTING.md)

# License

[GNU GPLv3](LICENSE.txt)

# Releases

(https://alus-builds.s3.eu-central-1.amazonaws.com/release)[https://alus-builds.s3.eu-central-1.amazonaws.com/release]
