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
#include "egm96_test.cuh"

#include "snap-dem/dem/dataio/earth_gravitational_model96.cuh"

namespace alus {
namespace tests {

__global__ void EGM96Tester(double* lats, double* lons, float* results, EGM96data data) {
    const int idx = threadIdx.x + (blockDim.x * blockIdx.x);

    if (idx < data.size) {
        results[idx] = snapengine::earthgravitationalmodel96computation::GetEGM96(lats[idx], lons[idx], data.max_lats,
                                                                                  data.max_lons, data.egm);
    }
}

cudaError_t LaunchEGM96(dim3 grid_size, dim3 block_size, double* lats, double* lons, float* results, EGM96data data) {
    EGM96Tester<<<grid_size, block_size>>>(lats, lons, results, data);
    return cudaGetLastError();
}

}  // namespace tests
}  // namespace alus
