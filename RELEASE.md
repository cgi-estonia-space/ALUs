# Release 1.6.0

## Breaking changes

## Known Caveats
* Coherence estimation results can have missing pixels(or coherence 0 values) on north and east direction on coastlines when compared to SNAP.
  Currently it is unresolved what is the correct way. See examples [A](https://alus-goods-set.s3.eu-central-1.amazonaws.com/alus_repo_docs/coh_missing_pixels.png) and [B](https://alus-goods-set.s3.eu-central-1.amazonaws.com/alus_repo_docs/beirut_iw1_b6_coastal.png).
  See posts about the issues/bugs - [tile size affecting results](https://forum.step.esa.int/t/tile-size-is-affecting-coregistration-results/32193/4)
  and [no-data value interpretation](https://forum.step.esa.int/t/coregistration-no-data-value/35304/2).
* Due to the nature of floating point arithmetic there are some discrepancies when compared to SNAP, see [slideshow](https://alus-goods-set.s3.eu-central-1.amazonaws.com/alus_repo_docs/GPU-GSTP-MPR-0008.pdf)

## Major Features and Improvements
* Basic S1 metadata in the produced results for `alus-coh`, `alus-cal` and `alus-coht` - [PR item](https://github.com/cgi-estonia-space/ALUs/pull/29)
* Polynomial estimation for subtract flat earth phase for each burst in coherence opeation - [PR item](https://github.com/cgi-estonia-space/ALUs/pull/26)
* Jupyter notebook and related dependencies updates - [PR item](https://github.com/cgi-estonia-space/ALUs/pull/25)
* GRD support for `alus-cal` - [PR item](https://github.com/cgi-estonia-space/ALUs/pull/31)

## Bug Fixes and Other Changes

## Thanks to our Contributors


# Release 1.5.0

## Breaking changes
* All the references and build automation functionality linked to Bitbucket is removed. Everything is now migrated to Github, hence
  Github actions are supported only.

## Known Caveats
* Coherence estimation results can have missing pixels(or coherence 0 values) on north and east direction on coastlines when compared to SNAP.
  Currently it is unresolved what is the correct way. See examples [A](https://alus-goods-set.s3.eu-central-1.amazonaws.com/alus_repo_docs/coh_missing_pixels.png) and [B](https://alus-goods-set.s3.eu-central-1.amazonaws.com/alus_repo_docs/beirut_iw1_b6_coastal.png).
  See posts about the issues/bugs - [tile size affecting results](https://forum.step.esa.int/t/tile-size-is-affecting-coregistration-results/32193/4)
  and [no-data value interpretation](https://forum.step.esa.int/t/coregistration-no-data-value/35304/2).
* Due to the nature of floating point arithmetic there are some discrepancies when compared to SNAP, see [slideshow](https://alus-goods-set.s3.eu-central-1.amazonaws.com/alus_repo_docs/GPU-GSTP-MPR-0008.pdf)
* Processing scenes with Copernicus DEM 30m COG has not well defined behavior for some specific cases.
  See workaround in ALUs - https://github.com/cgi-estonia-space/ALUs/wiki/DEM-handling-functionality-for-Sentinel-1-processors#copernicus-dem-cog-30m
  SNAP processing discrepancies - https://forum.step.esa.int/t/copernicus-dem-complications-when-coregistering-s1/38659/3

## Major Features and Improvements
* Shapefiles(.shp) are supported for specifying AOI - [PR item](https://github.com/cgi-estonia-space/ALUs/pull/17)
* Copernicus DEM 30m COG support - [PR1 item](https://github.com/cgi-estonia-space/ALUs/pull/19) [PR2 item](https://github.com/cgi-estonia-space/ALUs/pull/22)

## Bug Fixes and Other Changes
* 'Secret' log format can be invoked with `--log_format_creodias`. This enables JSON log output with 3 levels - DEBUG, INFO, ERROR - [PR item](https://github.com/cgi-estonia-space/ALUs/pull/14)
* 'alus-coh' will process scenes without orbit files now, before supplying orbit files was compulsory - [PR item](https://github.com/cgi-estonia-space/ALUs/pull/15)
* Some scenes had incorrect merge overlap for IW2 and IW3 for all S1 routines, this is now fixed and made faster - [PR item](https://github.com/cgi-estonia-space/ALUs/pull/20)

## Thanks to our Contributors


# Release 1.4.0

## Breaking changes

## Known Caveats
* Coherence estimation results can have missing pixels(or coherence 0 values) on north and east direction on coastlines when compared to SNAP.
  Currently it is unresolved what is the correct way. See examples [A](https://alus-goods-set.s3.eu-central-1.amazonaws.com/alus_repo_docs/coh_missing_pixels.png) and [B](https://alus-goods-set.s3.eu-central-1.amazonaws.com/alus_repo_docs/beirut_iw1_b6_coastal.png).
  See posts about the issues/bugs - [tile size affecting results](https://forum.step.esa.int/t/tile-size-is-affecting-coregistration-results/32193/4)
  and [no-data value interpretation](https://forum.step.esa.int/t/coregistration-no-data-value/35304/2).
* Due to the nature of floating point arithmetic there are some discrepancies when compared to SNAP, see [slideshow](https://alus-goods-set.s3.eu-central-1.amazonaws.com/alus_repo_docs/GPU-GSTP-MPR-0008.pdf)
* There are two code paths to calculate terrain corrected outputs' dimensions and geocoordinates
  ([see description](https://forum.step.esa.int/t/range-doppler-terrain-correction-raster-size-and-coordinates-inconsistency/35977)). Since ALUs
  implements the SNAP UI way the end results probably differ when compared to SNAP GPT outputs.

## Major Features and Improvements
* ALOS PALSAR Zero-Doppler L0 focuser implemented
* Subswath merge support for calibration routine

## Bug Fixes and Other Changes
* Tile access error in coherence operator fixed

## Thanks to our Contributors

# Release 1.3.0

## Breaking changes

## Known Caveats
* Coherence estimation results can have missing pixels(or coherence 0 values) on north and east direction on coastlines when compared to SNAP.
  Currently it is unresolved what is the correct way. See examples [A](https://alus-goods-set.s3.eu-central-1.amazonaws.com/alus_repo_docs/coh_missing_pixels.png) and [B](https://alus-goods-set.s3.eu-central-1.amazonaws.com/alus_repo_docs/beirut_iw1_b6_coastal.png).
  See posts about the issues/bugs - [tile size affecting results](https://forum.step.esa.int/t/tile-size-is-affecting-coregistration-results/32193/4)
  and [no-data value interpretation](https://forum.step.esa.int/t/coregistration-no-data-value/35304/2).
* Due to the nature of floating point arithmetic there are some discrepancies when compared to SNAP, see [slideshow](https://alus-goods-set.s3.eu-central-1.amazonaws.com/alus_repo_docs/GPU-GSTP-MPR-0008.pdf)
* There are two code paths to calculate terrain corrected outputs' dimensions and geocoordinates
  ([see description](https://forum.step.esa.int/t/range-doppler-terrain-correction-raster-size-and-coordinates-inconsistency/35977)). Since ALUs
  implements the SNAP UI way the end results probably differ when compared to SNAP GPT outputs.

## Major Features and Improvements
* Gabor feature extraction processor added
* Preliminary ALOS PALSAR level 0 focusser added

## Bug Fixes and Other Changes

## Thanks to our Contributors

# Release 1.2.0

## Breaking changes

## Known Caveats
* Coherence estimation results can have missing pixels(or coherence 0 values) on north and east direction on coastlines when compared to SNAP.
  Currently it is unresolved what is the correct way. See examples [A](https://alus-goods-set.s3.eu-central-1.amazonaws.com/alus_repo_docs/coh_missing_pixels.png) and [B](https://alus-goods-set.s3.eu-central-1.amazonaws.com/alus_repo_docs/beirut_iw1_b6_coastal.png).
  See posts about the issues/bugs - [tile size affecting results](https://forum.step.esa.int/t/tile-size-is-affecting-coregistration-results/32193/4) 
  and [no-data value interpretation](https://forum.step.esa.int/t/coregistration-no-data-value/35304/2).
* Due to the nature of floating point arithmetic there are some discrepancies when compared to SNAP, see [slideshow](https://alus-goods-set.s3.eu-central-1.amazonaws.com/alus_repo_docs/GPU-GSTP-MPR-0008.pdf)
* There are two code paths to calculate terrain corrected outputs' dimensions and geocoordinates
  ([see description](https://forum.step.esa.int/t/range-doppler-terrain-correction-raster-size-and-coordinates-inconsistency/35977)). Since ALUs
  implements the SNAP UI way the end results probably differ when compared to SNAP GPT outputs.

## Major Features and Improvements
* Thermal Noise Removal step added to calibration routine

## Bug Fixes and Other Changes

## Thanks to our Contributors



# Release 1.1.0

2 additional processors:
* `alus-resa` - resampling raster data
* `alus-coht` - coherence timeline generation

## Breaking changes

## Known Caveats
* Coherence estimation results can have missing pixels(or coherence 0 values) on north and east direction on coastlines when compared to SNAP.
  Currently it is unresolved what is the correct way. See examples [A](https://alus-goods-set.s3.eu-central-1.amazonaws.com/alus_repo_docs/coh_missing_pixels.png) and [B](https://alus-goods-set.s3.eu-central-1.amazonaws.com/alus_repo_docs/beirut_iw1_b6_coastal.png).
  See posts about the issues/bugs - [tile size affecting results](https://forum.step.esa.int/t/tile-size-is-affecting-coregistration-results/32193/4) 
  and [no-data value interpretation](https://forum.step.esa.int/t/coregistration-no-data-value/35304/2).
* Due to the nature of floating point arithmetic there are some discrepancies when compared to SNAP, see [slideshow](https://alus-goods-set.s3.eu-central-1.amazonaws.com/alus_repo_docs/GPU-GSTP-MPR-0008.pdf)

## Major Features and Improvements
* `--no_mask_cor` implemented to disable masking during the coregister step for coherence estimation. Disabling avoids 
  bug which creates incorrect pixels on coastline. Using similar functionality in SNAP would make results match on 
  coastlines 100%.
* Merge operation supported for coherence estimation. It is triggered by specifying AOI and no subswaths. This differs
  from SNAP where merge could be generated by manually specifying subswaths. Subswath specification functions only for 
  single subswath in ALUs. Also overlap areas in range direction between subswaths are more optimized where overlapping
  areas are minimized as much as possible.
* Orbit file directories without structure (all orbit files are stored with no hierarchy at a single level) are 
  supported for coherence estimation now.
* Resampling processor implemented. Specific support implemented for Sentinel-2 L1C and L2A datasets. Other rasters
  are resampled in a 'generic' way. Additional tiling and reprojection can be specified.

## Bug Fixes and Other Changes
* Memory leak bug fixed for coregistration operator.

## Thanks to our Contributors


# Release 1.0.0

This is the first release with refactored architecture where each processor is a different executable.

## Breaking changes
* Old commands do not work since the `alus` single executable is broken down into multiple separate executables

## Known Caveats
* Coherence estimation results can have missing pixels(or coherence 0 values) on north and east direction on coastlines when compared to SNAP.
  Currently it is unresolved what is the correct way. See examples [A](https://alus-goods-set.s3.eu-central-1.amazonaws.com/alus_repo_docs/coh_missing_pixels.png) and [B](https://alus-goods-set.s3.eu-central-1.amazonaws.com/alus_repo_docs/beirut_iw1_b6_coastal.png).
  See posts about the issues/bugs - [tile size affecting results](https://forum.step.esa.int/t/tile-size-is-affecting-coregistration-results/32193/4) and [no-data value interpretation](https://forum.step.esa.int/t/coregistration-no-data-value/35304/2).
* Due to the nature of floating point arithmetic there are some discrepancies when compared to SNAP, see [slideshow](https://alus-goods-set.s3.eu-central-1.amazonaws.com/alus_repo_docs/GPU-GSTP-MPR-0008.pdf)

## Major Features and Improvements
* There is a Jupyter notebook, which enables to more easily assign processing parameters, inputs and it automatically downloads needed auxiliary files background. 
  This should not be used for speed comparisons, because data transfers and unpacking ZIP archives would consume most of the total processing time.
* GPU device initialization and property component implemented - based on this functionality resources and parameters could be assigned better

## Bug Fixes and Other Changes
* Officially upgraded to Ubuntu 20.04 LTS
* Clang 10 compiler support added - refactored minor issues and CI pipeline checks added
* Boost library versions downgraded to 1.71 to match the distro default ones

## Thanks to our Contributors


# Release 0.9.0

This will be the last release with the current architecture (algorithms as shared libraries) and Ubuntu 18.04 as the officially supported platform.

## Breaking changes

## Known Caveats

* Auxiliary files must be separately downloaded and supplied via CLI arguments.
* Coherence estimation results can have missing pixels(or coherence 0 values) on north and east direction on coastlines.
  Currently investigation is ongoing to find out the exact reason. See examples [A](https://alus-goods-set.s3.eu-central-1.amazonaws.com/alus_repo_docs/coh_missing_pixels.png) and [B](https://alus-goods-set.s3.eu-central-1.amazonaws.com/alus_repo_docs/beirut_iw1_b6_coastal.png).

## Major Features and Improvements

* All the files and directories are formatted and named according to rules.
* License headers added where missing.
* Scripts added for automated code style and content conformity. This is checked for every pull request now.

## Bug Fixes and Other Changes

## Thanks to our Contributors 

# Release 0.8.0

## Breaking changes

## Known Caveats

* Auxiliary files must be separately downloaded and supplied via CLI arguments
* Coherence estimation results can have missing pixels(or coherence 0 values) on north and east direction on coastlines.
  Currently investigation is ongoing to find out the exact reason. See examples [A](https://alus-goods-set.s3.eu-central-1.amazonaws.com/alus_repo_docs/coh_missing_pixels.png) and [B](https://alus-goods-set.s3.eu-central-1.amazonaws.com/alus_repo_docs/beirut_iw1_b6_coastal.png)

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

