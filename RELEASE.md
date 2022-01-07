# Release 0.8.0

## Breaking changes

## Known Caveats

* Auxiliary files must be separately downloaded and supplied via CLI arguments
* Coherence estimation results can have missing pixels(or coherence 0 values) on north and east direction on coastlines.
  Currently investigation is ongoing to find out the exact reason. See examples [A](docs/coh_missing_pixels.png) and [B](docs/beirut_iw1_b6_coastal.png)

## Major Features and Improvements

* ZIP archive support
  * Sentinel 1 SLC scenes
  * SRTM3 DEM files
  * POEORB files

## Bug Fixes and Other Changes

## Thanks to our Contributors

# Release 0.7.0

[Pre release SNAP comparison](docs/SNAP_COMPARISON.md)

## Breaking changes

* Git LFS references removed
  * Replaced by custom `.alus-lfs` script in the root of the repository along with `alusresources` file
  * CI pipelines are now 2x and more faster

## Known Caveats

* ZIP files are not supported, any input datasets and auxiliary files must be extracted
* Auxiliary files must be separately downloaded and supplied via CLI arguments

## Major Features and Improvements

* Improvements for calibration routine:
  * Split arguments and operation supported and integrated, also support for burst index and AOI WKT polygon splitting
  * SRTM3 DEM loading is asynchronous
* Improvements for coherence routine:
  * Support for split operation by burst index or AOI WKT polygon
  * Performance optimizations for backgeocoding and coherence operations
    * Sin and Cos computations with single CUDA API call for backgeocoding
    * INT16 is integral type for raster data in backgeocoding now
    * Pinned memory and streams utilisation for coherence operator
    * Threading refactoring. Coregistration output now supports 4 datasets with 1 band or 1 dataset with 4 bands
    * Compute extended amount no longer reallocates metadata for each call and uses atomics for index calculations

## Bug Fixes and Other Changes

* Backgeocoding operator
  * Kernel block and grid parameters fixed which caused processing to fail on graphic cards with CC 7.5
* Coherence operator
  * Metadata mishandling caused wrong results for scenes with **near range on left**
  * Different tile sizes altering results is fixed
* Unnecessary shared libraries removed from release binary package
* More nightly test validation scenes
  * Also refactored the setup so that the scripts could be run on individual development environment
* SNAP-ENGINE and Sentinel 1 toolbox port:
  * relocated ported structures to separate folders in the root of the repo

## Thanks to our Contributors


# Release 0.6.0

[Slides about the speed improvements and accuracy](docs/GPU-GSTP-27_08_2021.pdf)  
[Version 0.6 binary](https://alus-builds.s3.eu-central-1.amazonaws.com/release/v0.6.0/alus-nightly-834-2021_09_14_1420.tar.gz)

## Breaking Changes

* Coherence operator and estimation routine:
    * Completely removed any TensorFlow implementation code
    * From scratch implementation using only core Nvidia native APIs(CUDA, cuBLAS).
    * Any included Tensorflow libraries/headers removed from repository

## Known Caveats

* No AOI or burst range based processing, only whole subswaths

## Major Features and Improvements

* Optimizations for Coherence estimation and Calibration routines:
    * All the operators use memory more efficiently (CUDA pinned memory, unified memory areas)
    * Speedup from 2x-10x achieved depending on the operator
    * Datasets of intermediate results are not written to disk by default
  
* Unified logging system implemented using Boost's Log.


## Bug Fixes and Other Changes

* SNAP-ENGINE port:
    * Some of the cyclic references that caused memory leaks were removed
      * Some still remain

* Coherence estimation routine
  * Polarization and subswath parameters' hardcoded values removed

* Backgeocoding
    * Out of bounds bug fixed
    * Triangular interpolation optimizations
    
* Calibration
    * Sigma0 and dn bands results' correction


## Thanks to our Contributors

