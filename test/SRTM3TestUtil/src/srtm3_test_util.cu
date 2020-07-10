#include "srtm3_test_util.cuh"

#include "srtm3_elevation_calc.cuh"

namespace alus {
namespace tests{

__global__ void SRTM3AltitudeTester(double *lats, double *lons, double *results, SRTM3TestData data){
    const int idx = threadIdx.x + (blockDim.x*blockIdx.x);

    if(idx < data.size){
        results[idx] = snapengine::srtm3elevationmodel::GetElevation(lats[idx], lons[idx], &data.tiles);
    }
}


cudaError_t LaunchSRTM3AltitudeTester(dim3 grid_size, dim3 block_size, double *lats, double *lons, double *results, SRTM3TestData data){

    SRTM3AltitudeTester<<<grid_size, block_size>>>(
        lats,
        lons,
        results,
        data
    );
    return cudaGetLastError();
}

}//namespace
}//namespace
