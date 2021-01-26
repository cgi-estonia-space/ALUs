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
#include "burst_offset_computation.h"

#include "backgeocoding_constants.h"
#include "position_data.h"

#include "backgeocoding_utils.cuh"
#include "geo_utils.cuh"
#include "math_utils.cuh"
#include "srtm3_elevation_calc.cuh"

namespace alus {
namespace backgeocoding {

__global__ void ComputeBurstOffsetKernel(BurstOffsetKernelArgs args) {
    const size_t index_x = threadIdx.x + blockDim.x * blockIdx.x;
    const size_t index_y = threadIdx.y + blockDim.y * blockIdx.y;

    if (index_x >= args.width || index_y >= args.height || *args.burst_offset != INVALID_BURST_OFFSET) {
        return;
    }
    const double latitude = args.latitudes[index_y * args.width + index_x];
    const double longitude = args.longitudes[index_x + index_y * args.width];

    const double altitude = snapengine::srtm3elevationmodel::GetElevation(latitude, longitude, &args.srtm3_tiles);
    if (altitude == snapengine::srtm3elevationmodel::NO_DATA_VALUE) {
        return;
    }

    s1tbx::PositionData position_data{};
    snapengine::geoutils::Geo2xyzWgs84Impl(latitude, longitude, altitude, position_data.earth_point);
    const BurstIndices master_burst_indices = GetBurstIndices(args.master_sentinel_utils,
                                                              args.master_subswath_info,
                                                              position_data.earth_point,
                                                              args.master_orbit,
                                                              args.master_num_orbit_vec,
                                                              args.master_dt);

    const BurstIndices slave_burst_indices = GetBurstIndices(args.slave_sentinel_utils,
                                                             args.slave_subswath_info,
                                                             position_data.earth_point,
                                                             args.slave_orbit,
                                                             args.slave_num_orbit_vec,
                                                             args.slave_dt);
    if (!master_burst_indices.valid || !slave_burst_indices.valid ||
        (master_burst_indices.first_burst_index == -1 && master_burst_indices.second_burst_index == -1) ||
        (slave_burst_indices.first_burst_index == -1 && slave_burst_indices.second_burst_index == -1)) {
        return;
    }
    int old = *args.burst_offset;
    bool execution_condition =
        master_burst_indices.in_upper_part_of_first_burst == slave_burst_indices.in_upper_part_of_first_burst;
    mathutils::Cas(&old,
                   INVALID_BURST_OFFSET * execution_condition,
                   slave_burst_indices.first_burst_index - master_burst_indices.first_burst_index);

    execution_condition = mathutils::Xor(
        execution_condition,
        slave_burst_indices.second_burst_index != -1 &&
            master_burst_indices.in_upper_part_of_first_burst == slave_burst_indices.in_upper_part_of_second_burst);
    mathutils::Cas(&old,
                   INVALID_BURST_OFFSET * execution_condition,
                   slave_burst_indices.second_burst_index - master_burst_indices.first_burst_index);

    execution_condition = mathutils::Xor(
        execution_condition,
        master_burst_indices.second_burst_index != -1 &&
            master_burst_indices.in_upper_part_of_second_burst == slave_burst_indices.in_upper_part_of_first_burst);
    mathutils::Cas(&old,
                   INVALID_BURST_OFFSET * execution_condition,
                   slave_burst_indices.first_burst_index - master_burst_indices.second_burst_index);

    execution_condition = mathutils::Xor(
        execution_condition,
        master_burst_indices.second_burst_index != -1 && slave_burst_indices.second_burst_index != -1 &&
            master_burst_indices.in_upper_part_of_second_burst == slave_burst_indices.in_upper_part_of_second_burst);
    mathutils::Cas(&old,
                   INVALID_BURST_OFFSET * execution_condition,
                   slave_burst_indices.second_burst_index - master_burst_indices.second_burst_index);
    atomicCAS(args.burst_offset, INVALID_BURST_OFFSET * (old != INVALID_BURST_OFFSET), old);
}

cudaError LaunchBurstOffsetKernel(BurstOffsetKernelArgs& args, int* burst_offset) {
    dim3 block_dim(24, 24);
    dim3 grid_dim(cuda::GetGridDim(block_dim.x, args.width), cuda::GetGridDim(block_dim.y, args.height));
    ComputeBurstOffsetKernel<<<grid_dim, block_dim>>>(args);
    cudaDeviceSynchronize();

    cudaError cuda_error = cudaGetLastError();

    CHECK_CUDA_ERR(cudaMemcpy(burst_offset, args.burst_offset, sizeof(int), cudaMemcpyDeviceToHost));
    return cuda_error;
}
}  // namespace backgeocoding
}  // namespace alus