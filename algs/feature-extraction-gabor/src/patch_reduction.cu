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

#include "patch_reduction.h"

namespace {
constexpr int BLOCK_X_SIZE = 32;
constexpr int BLOCK_Y_SIZE = 32;
constexpr int BLOCK_SIZE = BLOCK_X_SIZE * BLOCK_Y_SIZE;
}  // namespace

// Calculate mean for each sub 2D patch present in the src image, each patch has only 1 thread block operating on it
// This can be inefficient, if the total number of patches is small and/or no streams are used
__global__ void PatchMeanReduction(const float* src, float* result, int patch_size, int edge_size) {
    __shared__ float shared_acc[BLOCK_SIZE];

    const int n_x_patches = gridDim.x;
    const int pad_size = edge_size / 2;
    const int shared_idx = threadIdx.x + threadIdx.y * BLOCK_X_SIZE;
    const int result_idx = blockIdx.x + blockIdx.y * n_x_patches;

    // convert the grid index to src image coordinates
    const int padded_patch_size = patch_size + edge_size;
    const int src_width = padded_patch_size * n_x_patches;
    const int patch_start_x = blockIdx.x * padded_patch_size;
    const int patch_start_y = blockIdx.y * padded_patch_size;

    // 2D sum loop indexes
    const int start_x = patch_start_x + threadIdx.x + pad_size;
    const int start_y = patch_start_y + threadIdx.y + pad_size;
    const int last_x = patch_start_x + pad_size + patch_size;
    const int last_y = patch_start_y + pad_size + patch_size;

    // sum all pixels for this thread
    float acc = 0.0f;
    for (int y = start_y; y < last_y; y += BLOCK_Y_SIZE) {
        float tmp = 0.0f;
        for (int x = start_x; x < last_x; x += BLOCK_X_SIZE) {
            tmp += src[x + y * src_width];
        }
        acc += tmp;
    }

    // store each thread's result in shared memory, ensure all threads reach this point
    shared_acc[shared_idx] = acc;
    __syncthreads();

    // reduce the shared memory array to the first element
    // Reduction #3: Sequential Addressing from NVidia Optimizing Parallel Reduction
    for (int s = BLOCK_SIZE / 2; s > 0; s /= 2) {
        if (shared_idx < s) {
            shared_acc[shared_idx] += shared_acc[shared_idx + s];
        }
        __syncthreads();
    }

    // write the final result to global memory
    if (shared_idx == 0) {
        const float mean = shared_acc[0] / static_cast<float>(patch_size * patch_size);
        result[result_idx] = mean;
    }
}

// very similar to PatchMeanReduction
__global__ void PatchStdDevReduction(const float* src, const float* patch_means, float* result, int patch_size,
                                     int edge_size) {
    __shared__ float shared_acc[BLOCK_SIZE];

    const int n_x_patches = gridDim.x;
    const int pad_size = edge_size / 2;
    const int shared_idx = threadIdx.x + threadIdx.y * BLOCK_X_SIZE;
    const int result_idx = blockIdx.x + blockIdx.y * n_x_patches;

    // convert the grid index to src image coordinates
    const int padded_patch_size = patch_size + edge_size;
    const int src_width = padded_patch_size * n_x_patches;
    const int patch_start_x = blockIdx.x * padded_patch_size;
    const int patch_start_y = blockIdx.y * padded_patch_size;

    // 2D sum loop indexes
    const int start_x = patch_start_x + threadIdx.x + pad_size;
    const int start_y = patch_start_y + threadIdx.y + pad_size;
    const int last_x = patch_start_x + pad_size + patch_size;
    const int last_y = patch_start_y + pad_size + patch_size;

    const float mean = patch_means[result_idx];

    // calculate sum of squares of differences from mean
    float acc = 0.0f;
    for (int y = start_y; y < last_y; y += BLOCK_Y_SIZE) {
        float tmp = 0.0f;
        for (int x = start_x; x < last_x; x += BLOCK_Y_SIZE) {
            float val = src[x + y * src_width];
            float diff = val - mean;
            tmp += diff * diff;
        }
        acc += tmp;
    }

    // store each thread's result in shared memory, ensure all threads reach this point
    shared_acc[shared_idx] = acc;
    __syncthreads();

    // reduce the shared memory array to the first element
    for (int s = BLOCK_SIZE / 2; s > 0; s /= 2) {
        if (shared_idx < s) {
            shared_acc[shared_idx] += shared_acc[shared_idx + s];
        }
        __syncthreads();
    }

    // write the final result to global memory
    if (shared_idx == 0) {
        // -1 for sample instead of population spacing?
        const float variance = shared_acc[0] / static_cast<float>(patch_size * patch_size - 1);
        result[result_idx] = sqrtf(variance);
    }
}

namespace alus ::featurextractiongabor {

void LaunchPatchMeanReduction(const float* d_src, float* d_result, int patch_size, int edge_size, int n_x_patches,
                              int n_y_patches, cudaStream_t stream) {
    dim3 block_dim(BLOCK_X_SIZE, BLOCK_Y_SIZE);
    dim3 grid_dim(n_x_patches, n_y_patches);
    PatchMeanReduction<<<grid_dim, block_dim, 0, stream>>>(d_src, d_result, patch_size, edge_size);
}

void LaunchPatchStdDevReduction(const float* d_src, const float* d_patch_means, float* d_result, int patch_size,
                                int edge_size, int n_x_patches, int n_y_patches, cudaStream_t stream) {
    dim3 block_dim(BLOCK_X_SIZE, BLOCK_Y_SIZE);
    dim3 grid_dim(n_x_patches, n_y_patches);
    PatchStdDevReduction<<<grid_dim, block_dim, 0, stream>>>(d_src, d_patch_means, d_result, patch_size, edge_size);
}
}  // namespace alus::featurextractiongabor
