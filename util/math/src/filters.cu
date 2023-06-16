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

__global__ void RefinedLee(cuda::KernelArray<float> in, cuda::KernelArray<float> out, int width, int height,
                           int window, float no_data) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    int no_data_counter = 0;
    // Calculate the mean
    int half_window = window / 2;
    float mean = 0.0f;
    for (int i = -half_window; i <= half_window; i++) {
        for (int j = -half_window; j <= half_window; j++) {
            int nx = x + i;
            int ny = y + j;

            // Check bounds
            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                const auto value = in.array[nx + ny * width];
                if (value == no_data) {
                    no_data_counter++;
                } else {
                    mean += value;
                }
            }
        }
    }
    const auto win_sq = window * window;
    // If more than half of the pixels in the window are no_data.
    if (no_data_counter > win_sq / 2) {
        out.array[x + y * width] = in.array[x + y * width];
        return;
    }
    mean /= win_sq;

    // Calculate the variance
    float variance = 0.0f;
    for (int i = -half_window; i <= half_window; i++) {
        for (int j = -half_window; j <= half_window; j++) {
            int nx = x + i;
            int ny = y + j;

            // Check bounds
            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                float diff = in.array[nx + ny * width] - mean;
                variance += diff * diff;
            }
        }
    }
    variance /= win_sq;

    // Refined Lee
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