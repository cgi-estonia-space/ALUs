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
#include "burst_indices.h"

#include "backgeocoding_utils.cuh"

namespace alus {
namespace tests {
__global__ void ComputeBurstIndicesKernel(double line_time_interval,
                                          double wavelength,
                                          int num_of_bursts,
                                          const double *burst_first_line_times,
                                          const double *burst_last_line_times,
                                          snapengine::PosVector earth_point,
                                          snapengine::OrbitStateVectorComputation *orbit,
                                          const size_t num_orbit_vec,
                                          const double dt,
                                          backgeocoding::BurstIndices *indices) {
    const auto block_id = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
    auto const thread_id = block_id * (blockDim.x * blockDim.y * blockDim.z) + threadIdx.z * blockDim.x * blockDim.y +
                           threadIdx.y * blockDim.x + threadIdx.x;
    indices[thread_id] = backgeocoding::GetBurstIndices(line_time_interval,
                                                        wavelength,
                                                        num_of_bursts,
                                                        burst_first_line_times,
                                                        burst_last_line_times,
                                                        earth_point,
                                                        orbit,
                                                        num_orbit_vec,
                                                        dt);
}

cudaError_t LaunchComputeBurstIndicesKernel(double line_time_interval,
                                            double wavelength,
                                            int num_of_bursts,
                                            const double *burst_first_line_times,
                                            const double *burst_last_line_times,
                                            snapengine::PosVector earth_point,
                                            snapengine::OrbitStateVectorComputation *orbit,
                                            const size_t num_orbit_vec,
                                            const double dt,
                                            backgeocoding::BurstIndices *indices,
                                            dim3 grid_dim,
                                            dim3 block_dim) {
    ComputeBurstIndicesKernel<<<grid_dim, block_dim>>>(line_time_interval,
                                        wavelength,
                                        num_of_bursts,
                                        burst_first_line_times,
                                        burst_last_line_times,
                                        earth_point,
                                        orbit,
                                        num_orbit_vec,
                                        dt,
                                        indices);
    auto error = cudaGetLastError();
    CHECK_CUDA_ERR(cudaDeviceSynchronize());

    return error;
}
}  // namespace tests
}  // namespace alus
