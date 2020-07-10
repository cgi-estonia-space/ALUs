#include "srtm3_formatter.cuh"
#include "earth_gravitational_model96.cuh"

#include <cstdio>

namespace alus {
namespace snapengine{

__global__ void FormatSRTM3dem(double *target, double *source, DemFormatterData data){
    const int idx = threadIdx.x + (blockDim.x*blockIdx.x);
    const int idy = threadIdx.y + (blockDim.y*blockIdx.y);
    double geo_pos_lon, geo_pos_lat, source_value;


    //possible bug:it is possible that this is a snap bug, as snap reads line 2501 when index is 2500.
    if(idx < data.x_size && idy < (data.y_size -1)){
        source_value = source[idx + data.x_size *(idy+1)];
        if(source_value != data.no_data_value){
            //everything that TileGeoReferencing.getGeoPos does.
            geo_pos_lon = data.m00*(idx + 0.5) + data.m01*(idy + 0.5) + data.m02;
            geo_pos_lat = data.m10*(idx + 0.5) + data.m11*(idy + 0.5) + data.m12;
            target[idx + data.x_size *idy] =
                source_value + snapengine::earthgravitationalmodel96::GetEGM96(
                                   geo_pos_lat,
                                   geo_pos_lon,
                                   data.max_lats,
                                   data.max_lons,
                                   data.egm);
        }else{
            target[idx + data.x_size *idy] = source_value;
        }

    }
}


cudaError_t LaunchDemFormatter(dim3 grid_size, dim3 block_size, double *target, double *source, DemFormatterData data){
    FormatSRTM3dem<<<grid_size, block_size>>>(target, source, data);
    return cudaGetLastError();
}

}//namespace
}//namespace
