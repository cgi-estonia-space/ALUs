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

#include "calc_kernels.cuh"

#include <cmath>

namespace alus::math::calckernels {

__global__ void CalcDb(cuda::KernelArray<float> buffer, size_t w, size_t h, float no_data_value) {
    const auto thread_x = threadIdx.x + blockIdx.x * blockDim.x;
    const auto thread_y = threadIdx.y + blockIdx.y * blockDim.y;

    if (thread_x >= w || thread_y >= h) {
        return;
    }

    const auto index = thread_y * w + thread_x;
    const auto orig_value = buffer.array[index];
    if (orig_value == 0 || isnan(orig_value)) {
        return;
    }
    if (!isnan(no_data_value) && orig_value == no_data_value) {
        return;
    }

    buffer.array[index] = log10(orig_value);
}

}  // namespace alus::math::calckernels