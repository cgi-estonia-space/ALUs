#include "srtm3_test_util.cuh"

#include "srtm3_elevation_calc.cuh"

namespace alus {
namespace tests{

__global__ void SRTM3AltitudeTester(double *lats, double *lons, double *results, SRTM3TestData data){
    const int idx = threadIdx.x + (blockDim.x*blockIdx.x);

    if(idx < data.size){
        results[idx] = snapengine::srtm3elevationmodel::getElevation(
            lats[idx],
            lons[idx],
            &data.tiles
        );
    }
}


cudaError_t launchSRTM3AltitudeTester(dim3 gridSize, dim3 blockSize, double *lats, double *lons, double *results, SRTM3TestData data){

    SRTM3AltitudeTester<<<gridSize, blockSize>>>(
        lats,
        lons,
        results,
        data
    );
    return cudaGetLastError();
}

}//namespace
}//namespace
