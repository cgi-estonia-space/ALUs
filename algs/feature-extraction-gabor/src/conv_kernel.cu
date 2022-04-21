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

#include "conv_kernel.h"

// 2D convolution of image patches, each thread calculates one pixel for the output
// the padding pixels used are zeroed in the result image, for easier comparison with the original C# code
// kernel array must be reversed before this call
__global__ void Conv2DPatchNaive(const float* src, float* dst, int width, int height, int patch_size,
                                 const float* kernel, int kernel_size) {
    const int x = threadIdx.x + blockDim.x * blockIdx.x;
    const int y = threadIdx.y + blockDim.y * blockIdx.y;

    const int padded_patch_size = kernel_size + patch_size;

    if (x >= width || y >= height) {
        return;
    }

    // find the patch this thread is in
    const int nth_patch_x = x / padded_patch_size;  // TODO get rid of integer division?
    const int nth_patch_y = y / padded_patch_size;
    const int patch_x_start = nth_patch_x * padded_patch_size;
    const int patch_y_start = nth_patch_y * padded_patch_size;
    const int patch_x_end = patch_x_start + padded_patch_size;
    const int patch_y_end = patch_y_start + padded_patch_size;

    const int dest_idx = x + y * width;
    const int pad_size = kernel_size / 2;

    const int start_x = x - pad_size;
    const int start_y = y - pad_size;

    if (x < patch_x_start + pad_size || x >= patch_x_end - pad_size || y < patch_y_start + pad_size ||
        y >= patch_y_end - pad_size) {
        // zero padding bytes
        dst[dest_idx] = 0.0f;
        return;
    }

    // convolution calculation
    float acc = 0.0f;
    for (int k_y = 0; k_y < kernel_size; k_y++) {
        const int y_idx = k_y + start_y;
        for (int k_x = 0; k_x < kernel_size; k_x++) {
            float mask_val = kernel[k_x + k_y * kernel_size];
            float src_pixel = src[start_x + k_x + y_idx * width];
            acc += mask_val * src_pixel;
        }
    }
    dst[dest_idx] = acc;
}

namespace alus::featurextractiongabor {
void LaunchConvKernel(const float* d_src, int width, int height, int patch_size, float* d_dest, const float* d_kernel,
                      int kernel_size, cudaStream_t stream) {
    dim3 block_dim{16, 16};
    dim3 grid_dim((width + 15) / 16, (height + 15) / 16);
    Conv2DPatchNaive<<<grid_dim, block_dim, 0, stream>>>(d_src, d_dest, width, height, patch_size, d_kernel,
                                                         kernel_size);
}
}  // namespace alus::featurextractiongabor