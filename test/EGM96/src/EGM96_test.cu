#include "EGM96_test.cuh"

#include "earth_gravitational_model96.cuh"

namespace slap{
namespace tests{

__global__ void EGM96Tester(double *lats, double *lons, float *results, EGM96data data){
    const int idx = threadIdx.x + (blockDim.x*blockIdx.x);

    if(idx < data.size){
        results[idx] = snapengine::earthgravitationalmodel96::getEGM96(lats[idx],lons[idx], data.MAX_LATS, data.MAX_LONS, data.egm);
    }
}


cudaError_t launchEGM96(dim3 gridSize, dim3 blockSize, double *lats, double *lons, float *results, EGM96data data){

    EGM96Tester<<<gridSize, blockSize>>>(
        lats,
        lons,
        results,
        data
    );
    return cudaGetLastError();
}

}//namespace
}//namespace
