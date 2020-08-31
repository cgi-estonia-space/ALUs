#pragma once

#include <cuda_runtime.h>

namespace alus {
namespace tests{

struct EGM96data{
    int max_lats;
    int max_lons;
    float* egm;
    int size;
};

cudaError_t LaunchEGM96(dim3 grid_size, dim3 block_size, double *lats, double *lons, float *results, EGM96data data);


}//namespace
}//namespace
