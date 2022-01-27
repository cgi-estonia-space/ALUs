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
#include "sar_geocoding_test.cuh"

#include "s1tbx-commons/sar_geocoding.cuh"

namespace alus {
namespace tests {

__global__ void ZeroDopplerTimeTestImpl(double* results, ZeroDopplerTimeData data) {
    const int idx = threadIdx.x + (blockDim.x * blockIdx.x);

    if (idx < data.data_size) {
        results[idx] = alus::s1tbx::sargeocoding::GetZeroDopplerTime(
            data.device_line_time_interval[idx], data.device_wavelengths[idx], data.device_earth_points[idx],
            data.orbit, data.num_orbit_vec, data.dt);
    }
}

cudaError_t LaunchZeroDopplerTimeTest(dim3 grid_size, dim3 block_size, double* results, ZeroDopplerTimeData data) {
    ZeroDopplerTimeTestImpl<<<grid_size, block_size>>>(results, data);
    return cudaGetLastError();
}

}  // namespace tests
}  // namespace alus