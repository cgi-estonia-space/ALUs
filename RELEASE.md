# Release 0.6.0

[Slides about the speed improvements and accuracy](docs/GPU-GSTP-27_08_2021.pdf)

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

