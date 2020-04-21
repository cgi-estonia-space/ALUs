# alus

The go to software for SAR Image GPU processing.

# build
I use the following script to build:

```
#!/bin/bash

build_dir=build

if [[ "$*" == "-c" ]] ; then
	rm -rf $build_dir
	echo "removing build dir"
else
	echo "not removing build dir"
fi

cmake -H. -B$build_dir
cmake --build $build_dir -j 3
```

# testing

I use the following script to run tests:

```
#!/bin/bash

cd build/test
./unit_test
cd ../..

```
