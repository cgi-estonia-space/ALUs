# Build steps #

Tensorflow version 2.4.1 with CUDA CC 5.0,6.0,7.0,7.5

* Run alus-infra docker
* git clone https://github.com/FloopCZ/tensorflow_cc.git (during this master branch with hashfeb418b was used)
* Can define project version in tensorflow_cc/PROJECT_VERSION using default one - 2.4.1
* Check tensorflow_cc/Dockerfiles - basically tried to install requirements specified there
* apt-get -y update 
* apt-get -y install
* apt install python3-dev
* Other ones are satisfied already in the container
* Download or use existing DEV!!! debian package for cudnn - libcudnn7-dev_7.6.5.32-1+cuda10.2_amd64.deb
* Download here - https://developer.nvidia.com/rdp/cudnn-archive (cuDNN Developer Library for Ubuntu18.04 (Deb))
* docker cp libcudnn7-dev_7.6.5.32-1+cuda10.2_amd64.deb priceless_saha:/root/
* apt install -y ~/libcudnn7-dev_7.6.5.32-1+cuda10.2_amd64.deb
* bazel_installer=bazel-3.1.0-installer-linux-x86_64.sh (tensorflow_cc default version - it says newer ones sometimes will fail building)
* apt install -y wget
* wget -P /tmp https://github.com/bazelbuild/bazel/releases/download/3.1.0/${bazel_installer} /tmp/bazel-3.1.0-installer-linux-x86_64.sh
* vim tensorflow_cc/cmake/build_tensorflow.sh.in
* Add supported compute capabilities there (5.0,6.0,7.0,7.5)
* cd tensorflow_cc
* make build
* cd build
* cmake -DLOCAL_RAM_RESOURCES=2048 ..
* link /usr/bin/python3 /usr/bin/python (Bazel 3.1.0 has a bug where it by default uses 'python')
* make
* modify install path by setting "CMAKE_INSTALL_PREFIX" in the top of cmake_install.cmake 
* make install
* docker cp priceless_saha:/root/tensorflow_cc_install /home/sven/alus/tensorflow/
* Create tensorflow platform folder - for example tensorflow_cc
* Create include and lib folders inside that
* Transfer built libraries
* cp tensorflow_cc_install/lib/* tensorflow_cc/lib/.
* Create additional symlinks because it has embedded dependency on shread libraries to loaded
* cd tensorflow_cc/lib
* ln -s libtensorflow_cc.so. libtensorflow_cc.so
* ln -s libtensorflow_cc.so. libtensorflow_cc.so.2 
* Following folders are needed in include:
* Copy them from "tensorflow_cc_install/include/tensorflow/bazel-bin"
* Also need to complement #define ABSL_HAVE_STD_STRING_VIEW 1 in tensorflow/include/absl/base/config.h
