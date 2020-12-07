# ALUS facility

Scripts that help in processing SAR images

## dem_supply.sh

### Requirements
One needs to install python tool **elevation**. For this project slightly modified version of this one at
https://github.com/svenKautlenbach/elevation should be downloaded. This project require **.ftw** files which
are otherwise not used in the original project.
Install:
```
# Clone or download .zip and unpack it
cd elevation
python3 -m pip install -r requirements-dev.txt
sudo -H python3 -m pip install -e .
```

### Usage

Supply input data product as argument - it determines the bounds of the scene and downloads needed
DEM files (if needed) using tool **eio**. Supported are SAFE archives (unpacked or zipped) and DIMAP
products (.dim should be supplied)
Script outputs DEM file list as a last line which can be automatically supplied to **alus** executable.

```
./dem_supply.sh S1A_3432_4324...324.dim
alus -input in.tif -output out.tif .... --dem "$(cat /tmp/elevation/log.txt | tail -n 1)"
#or
dem_files=`./dem_supply.sh S1A_3432_4324...324.dim | tail -n 1`
alus -input in.tif -output out.tif .... --dem "$dem_files"
```
