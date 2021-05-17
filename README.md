# ALUS

ALUS means beer in latvian.
Here it means Acceleration Library Used for Sentinel.
A software project that targets to utilize Nvidia GPUs for processing earth observation data (faster).

Kickstart of this project was funded through ESA's EOEP programme - http://www.esa.int/About_Us/Business_with_ESA/Business_Opportunities/Earth_Observation_Envelope_Programme
\
Developed by CGI Estonia.

# Dependencies

[Setup of dependencies](DEPENDENCIES.md)

# Building

```
cmake . -Bbuild
cd build
make -j8
```

# Executing

Main executable with algorithm shared libraries(.so) is located at **<build_dir>/alus_package**.\
In order to load shared library components one should move to directory where binaries are located or\
``LD_LIBRARY_PATH=$LD_LIBRARY_PATH:...<build_dir>/alus_package; export LD_LIBRARY_PATH``\
or move/create symlinks to a location that is present in default ``LD_LIBRARY_PATH``.

See ``--help`` for specific arguments how to invoke processing.

# Contributing

[Contribution guidelines](CONTRIBUTING.md)

# License

[GNU GPLv3](LICENSE.txt)
