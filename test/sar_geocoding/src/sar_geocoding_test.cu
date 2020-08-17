#include "sar_geocoding_test.cuh"

#include "sar_geocoding.cuh"

namespace alus {
namespace tests{

__global__ void ZeroDopplerTimeTestImpl(double *results, ZeroDopplerTimeData data){
    const int idx = threadIdx.x + (blockDim.x*blockIdx.x);

    if(idx < data.data_size){
        results[idx] = alus::s1tbx::sargeocoding::GetZeroDopplerTime(data.device_line_time_interval[idx],
                                                                     data.device_wavelengths[idx],
                                                                     data.device_earth_points[idx],
                                                                     data.orbit,
                                                                     data.num_orbit_vec,
                                                                     data.dt);
    }
}

cudaError_t LaunchZeroDopplerTimeTest(dim3 grid_size, dim3 block_size, double *results, ZeroDopplerTimeData data){

    ZeroDopplerTimeTestImpl<<<grid_size, block_size>>>(results, data);
    return cudaGetLastError();
}

}//namespace
}//namespace