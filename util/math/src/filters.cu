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

#include "filters.cuh"

#include <cmath>

namespace alus::math::filters {

__global__ void RefinedLee(cuda::KernelArray<float> in, cuda::KernelArray<float> out, int width, int height, int window,
                           float variance) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    int half_window = window / 2;
    float sum = 0.0f;
    float weights = 0.0f;

    for (int i = -half_window; i <= half_window; i++) {
        for (int j = -half_window; j <= half_window; j++) {
            int nx = x + i;
            int ny = y + j;

            if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;

            float diff = in.array[nx + ny * width] - in.array[x + y * width];
            float weight = expf(-diff * diff / (variance * variance));

            sum += in.array[nx + ny * width] * weight;
            weights += weight;
        }
    }

    out.array[x + y * width] = sum / weights;
}

}  // namespace alus::math::filters