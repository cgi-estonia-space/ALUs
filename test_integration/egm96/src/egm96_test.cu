#include "egm96_test.cuh"

#include "earth_gravitational_model96.cuh"

namespace alus {
namespace tests{

__global__ void EGM96Tester(double *lats, double *lons, float *results, EGM96data data){
    const int idx = threadIdx.x + (blockDim.x*blockIdx.x);

    if(idx < data.size){
        results[idx] = snapengine::earthgravitationalmodel96computation::GetEGM96(
            lats[idx], lons[idx], data.max_lats, data.max_lons, data.egm);
    }
}


cudaError_t LaunchEGM96(dim3 grid_size, dim3 block_size, double *lats, double *lons, float *results, EGM96data data){

    EGM96Tester<<<grid_size, block_size>>>(
        lats,
        lons,
        results,
        data
    );
    return cudaGetLastError();
}

}//namespace
}//namespace
