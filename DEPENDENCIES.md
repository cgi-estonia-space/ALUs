# CUDA support

All of the platforms listed below either for development/building or running require CUDA SDK version 10.2.

Possible compatibility with newer versions, but not tested/supported.

Nvidia driver version at least 440.33.

# GDAL

GDAL version 3.x is required for development/building or running.

# Development

In order to build or develop code platforms' dependency installation manual is provided below.

Officially Ubuntu 18.04 is supported.

Compatible docker image is available on [dockerhub](https://hub.docker.com/repository/docker/cgialus/alus-devel).

Development team has used also other flavor of Ubuntu which setup is also listed, but is not officially supported.

## Ubuntu 18.04


* ``apt-get update``
* ``apt install git git-lfs g++ gdb clang-format valgrind libssl-dev libgfortran3 libeigen3-dev zlib1g-dev zlib1g``
* ``apt-add-repository -y ppa:mhier/libboost-latest``
* ``apt install -y libboost1.74 libboost1.74-dev``
* CMake (version atleast 3.18 is required which is not available in default repos)
  * ``apt install software-properties-common``
  * ``wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | sudo apt-key add -``
  * ``apt-add-repository 'deb https://apt.kitware.com/ubuntu/ bionic main'``
  * ``apt install cmake``
  
* GDAL (must be self compiled and installed for version 3)
  * Download PROJ version 6.3 - https://download.osgeo.org/proj/proj-6.3.0.tar.gz
    * Follow installation manual provided in the archive
  * Download GDAL version 3.0.4 - https://github.com/OSGeo/gdal/releases/download/v3.0.4/gdal-3.0.4.tar.gz
    * Follow installation manual provided in the archive

## Pop_OS! 20.04 (Possible to convert to plain Ubuntu 20.04)

* ``apt install system76-cuda-10.2``
* ``export PATH=$HOME/bin:/usr/local/bin:/usr/lib/cuda/bin:$PATH``
* ``export LD_LIBRARY_PATH=/usr/lib/cuda/lib64:/usr/local/lib/:$LD_LIBRARY_PATH``
* ``update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-8 8``
* ``update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-8 8``
* ``apt install git-lfs gdal-bin libeigen3-dev python3-pip gdb clang-format valgrind``
* ``apt-add-repository -y ppa:mhier/libboost-latest``
* ``apt remove libboost1.71-dev``
* ``apt install -y libboost1.74 libboost1.74-dev``
* ``apt install libgdal-dev`` - sequence is important here since it requires libboost
* CMake (version atleast 3.18 is required which is not available in default repos)
  * ``apt install software-properties-common``
  * ``wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | sudo apt-key add -``
  * ``apt-add-repository 'deb https://apt.kitware.com/ubuntu/ focal main'``
  * ``apt install cmake``
