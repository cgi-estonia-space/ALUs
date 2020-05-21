#pragma once

#include <cuda_runtime.h>

namespace slap{
namespace tests{

struct EGM96data{
    int MAX_LATS;
    int MAX_LONS;
    double* egm;
    int size;
};

cudaError_t launchEGM96(dim3 gridSize, dim3 blockSize, double *lats, double *lons, float *results, EGM96data data);


}//namespace
}//namespace