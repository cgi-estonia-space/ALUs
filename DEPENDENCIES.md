# CUDA support

CUDA SDK version atleast 11.2.
Possible compatibility with newer/older versions, but not tested/supported.
Nvidia driver version at least 450.80.02.

# GDAL

GDAL version 3.x is required for development/building or running.

# Development

In order to build or develop code platforms' dependency installation manual is provided below.

Officially Ubuntu 20.04 is supported. May work on other versions, please see main CMakeLists.txt file for Boost, GDAL, CUDA SDK versions targeted.

Compatible docker image is available on [dockerhub](https://hub.docker.com/repository/docker/cgialus/alus-devel).

## Ubuntu 20.04

Note that for installing drivers a X(Desktop environment) system must be turned off. 

* ``apt update``
* ``apt install -y gcc g++``
* ``wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin``
* ``mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600``
* ``apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub``
* ``add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"``
* ``apt update``
* ``apt -y install cuda-11-2``
* ``echo "" >> .bashrc``
* ``echo "export LD_LIBRARY_PATH=/usr/local/cuda/lib64/:$LD_LIBRARY_PATH" >> ~/.bashrc``
* ``echo "export PATH=/usr/local/cuda/bin/:$PATH" >> ~/.bashrc``

For running only

* ``apt update && sudo apt install -y --no-install-recommends gdal-bin libgdal-dev libboost-program-options-dev libboost-date-time-dev libboost-iostreams-dev libboost-log-dev zlib1g-dev``

For running python supplementary tools

* ``apt install -y python3-venv pip python3-gdal``

For development

* ``apt update && sudo curl https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | sudo apt-key add - && sudo apt-add-repository -y 'deb https://apt.kitware.com/ubuntu/ focal main' && sudo apt update && sudo apt install -y cmake --no-install-recommends``
* ``apt install clang clang-format-10 clang-tidy-10 libeigen3-dev unzip``

## Pop_OS! 20.04

Core development team uses this flavor since it includes drivers and CUDA SDK repos are built in

* ``apt install system76-cuda-11.2``
* ``export LD_LIBRARY_PATH=/usr/lib/cuda/lib64:/usr/local/lib/:$LD_LIBRARY_PATH``

Development tools

* ``sudo apt update && sudo apt install -y --no-install-recommends software-properties-common curl ca-certificates && sudo curl https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | sudo apt-key add - && sudo apt-add-repository -y 'deb https://apt.kitware.com/ubuntu/ focal main' && sudo apt-get update && sudo apt install -y cmake --no-install-recommends && sudo apt-get clean``
* ``sudo apt-get update && sudo apt-get install -y --no-install-recommends clang unzip gdal-bin libgdal-dev libboost-program-options-dev libboost-date-time-dev libboost-iostreams-dev libboost-log-dev libeigen3-dev zlib1g-dev clang-format clang-tidy wget && sudo apt-get -y autoremove && sudo apt-get clean``

For running python supplementary tools

* ``apt install -y python3-venv pip``
