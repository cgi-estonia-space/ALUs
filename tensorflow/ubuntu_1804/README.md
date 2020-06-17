Version 2.1

Built on eOS .....
`bazel build --noincompatible_do_not_split_linking_cmdline tensorflow:libtensorflow_cc.so`

following configuration
```
build --action_env PYTHON_BIN_PATH="/usr/bin/python3"
build --action_env PYTHON_LIB_PATH="/usr/lib/python3/dist-packages"
build --python_path="/usr/bin/python3"
build:xla --define with_xla_support=true
build --action_env CUDA_TOOLKIT_PATH="/usr/local/cuda"
build --action_env TF_CUDA_COMPUTE_CAPABILITIES="5.0,6.0"
build --action_env LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/local/lib/:/usr/local/cuda/lib64:/usr/local/lib/:"
build --action_env GCC_HOST_COMPILER_PATH="/usr/bin/x86_64-linux-gnu-gcc-7"
build --config=cuda
build:opt --copt=-march=native
build:opt --copt=-Wno-sign-compare
build:opt --host_copt=-march=native
build:opt --define with_default_optimizations=true
test --flaky_test_attempts=3
test --test_size_filters=small,medium
test:v1 --test_tag_filters=-benchmark-test,-no_oss,-gpu,-oss_serial
test:v1 --build_tag_filters=-benchmark-test,-no_oss,-gpu
test:v2 --test_tag_filters=-benchmark-test,-no_oss,-gpu,-oss_serial,-v1only
test:v2 --build_tag_filters=-benchmark-test,-no_oss,-gpu,-v1only
build --action_env TF_CONFIGURE_IOS="0"
```
