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

#include "backgeocoding_constants.h"
#include "cuda_util.h"
#include "dem_calc.cuh"
#include "dem_property.h"
#include "elevation_mask_computation.h"

namespace alus {
namespace backgeocoding {

// regardless whether you mask or not, you need to run the if(data.device_x_points[idx] ... to purify the indexes.
__global__ void ElevationMask(ElevationMaskData data) {
    const size_t idx = threadIdx.x + (blockDim.x * blockIdx.x);

    if (idx < data.size) {
        if (data.device_x_points[idx] == INVALID_INDEX || data.device_y_points[idx] == INVALID_INDEX) {
            data.device_x_points[idx] = INVALID_INDEX;
            data.device_y_points[idx] = INVALID_INDEX;
            return;
        }

        const double lat = data.device_lat_array[idx];
        const double lon = data.device_lon_array[idx];
        const double alt = snapengine::dem::GetElevation(lat, lon, &data.tiles, data.dem_property);

        // TODO: this may need to change if we decide not to use mask.
        if (data.mask_out_area_without_elevation && alt == snapengine::srtm3elevationmodel::NO_DATA_VALUE) {
            data.device_x_points[idx] = INVALID_INDEX;
            data.device_y_points[idx] = INVALID_INDEX;
        } else {
            (*data.not_null_counter)++;  // race condition is not important here.
        }
    }
}

cudaError_t LaunchElevationMask(ElevationMaskData data) {
    dim3 block_size(416);
    dim3 grid_size(cuda::GetGridDim(block_size.x, data.size));

    ElevationMask<<<grid_size, block_size>>>(data);
    return cudaGetLastError();
}

}  // namespace backgeocoding
}  // namespace alus
