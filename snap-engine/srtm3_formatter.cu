/**
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the Free
 * Software Foundation; either version 3 of the License, or (at your option)
 * any later version.
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for
 * more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, see http://www.gnu.org/licenses/
 */
#include "srtm3_format_computation.h"

#include "snap-dem/dem/dataio/earth_gravitational_model96.cuh"

namespace alus {
namespace snapengine{

__global__ void FormatSRTM3dem(float *target, float *source, Srtm3FormatComputation data){
    const int idx = threadIdx.x + (blockDim.x*blockIdx.x);
    const int idy = threadIdx.y + (blockDim.y*blockIdx.y);
    double geo_pos_lon, geo_pos_lat;
    float source_value;


    if(idx < data.x_size && idy < data.y_size){
        source_value = source[idx + data.x_size * idy];
        if(source_value != data.no_data_value){
            //everything that TileGeoReferencing.getGeoPos does.
            geo_pos_lon = data.m00*(idx + 0.5) + data.m01*(idy + 0.5) + data.m02;
            geo_pos_lat = data.m10*(idx + 0.5) + data.m11*(idy + 0.5) + data.m12;
            target[idx + data.x_size *idy] =
                source_value + snapengine::earthgravitationalmodel96computation::GetEGM96(
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


cudaError_t LaunchDemFormatter(dim3 grid_size, dim3 block_size, float *target, float *source, Srtm3FormatComputation data){
    FormatSRTM3dem<<<grid_size, block_size>>>(target, source, data);
    cudaDeviceSynchronize();
    return cudaGetLastError();
}

}//namespace
}//namespace
