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

#include "dem_egm96.h"

#include "cuda_util.h"
#include "snap-dem/dem/dataio/earth_gravitational_model96.cuh"

namespace {

__global__ void ConditionKernel(float* target, float* source, alus::dem::EgmFormatProperties tile_prop) {
    const int idx = threadIdx.x + (blockDim.x * blockIdx.x);
    const int idy = threadIdx.y + (blockDim.y * blockIdx.y);
    double geo_pos_lon, geo_pos_lat;
    float source_value;

    if (idx < tile_prop.tile_size_x && idy < tile_prop.tile_size_y) {
        source_value = source[idx + tile_prop.tile_size_x * idy];
        if (source_value != tile_prop.no_data_value) {
            // everything that TileGeoReferencing.getGeoPos does.
            geo_pos_lon = tile_prop.m00 * (idx + 0.5) + tile_prop.m01 * (idy + 0.5) + tile_prop.m02;
            geo_pos_lat = tile_prop.m10 * (idx + 0.5) + tile_prop.m11 * (idy + 0.5) + tile_prop.m12;
            target[idx + tile_prop.tile_size_x * idy] =
                source_value + alus::snapengine::earthgravitationalmodel96computation::GetEGM96(
                                   geo_pos_lat, geo_pos_lon, tile_prop.grid_max_lat, tile_prop.grid_max_lon,
                                   tile_prop.device_egm_array);
        } else {
            target[idx + tile_prop.tile_size_x * idy] = source_value;
        }
    }
}
}  // namespace

namespace alus::dem {

void ConditionWithEgm96(dim3 grid_size, dim3 block_size, float* target, float* source, EgmFormatProperties prop) {
    ConditionKernel<<<grid_size, block_size>>>(target, source, prop);
    CHECK_CUDA_ERR(cudaDeviceSynchronize());
    CHECK_CUDA_ERR(cudaGetLastError());
}

}  // namespace alus::dem