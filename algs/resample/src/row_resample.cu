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

#include "row_resample.h"

#include <cmath>

#include "cuda_ptr.h"
#include "cuda_util.h"

namespace {

__host__ __device__ inline void InterpolatePixelValue(int x, const float* in, int in_size, float* out, int out_size,
                                                      double pixel_ratio) {
    float dummy;
    // Division could be optimized, but not a showstopper now. Trying to preserve maximum amount of float precision.
    const auto start_dist = (x * in_size) / static_cast<double>(out_size);
    const auto end_dist = ((x + 1) * in_size) / static_cast<double>(out_size);
    const auto in_index1 = static_cast<int>(start_dist);  // Floor it.
    const auto in_index2 = static_cast<int>(end_dist);
    double index1_factor;
    double index2_factor;
    if (in_index1 == in_index2) {
        index1_factor = 1.0;
        index2_factor = 0.0;
    } else {
        float fract1 = std::modf(start_dist, &dummy);
        float fract2 = std::modf(end_dist, &dummy);
        index1_factor = (1 - fract1) / pixel_ratio;
        index2_factor = fract2 / pixel_ratio;
    }

    auto val1 = in[in_index1];
    auto val2 = in[in_index2];

    out[x] = val1 * index1_factor + val2 * index2_factor;
}

__global__ void ResampleKernel(const float* in, alus::RasterDimension in_dim, float* out, alus::RasterDimension out_dim,
                               double pixel_ratio) {
    const int idx = threadIdx.x + (blockDim.x * blockIdx.x);
    const int idy = threadIdx.y + (blockDim.y * blockIdx.y);

    if (idx >= out_dim.columnsX || idy >= out_dim.rowsY) {
        return;
    }

    const auto buffer_out_offset = idy * out_dim.columnsX;
    const auto buffer_in_offset = idy * in_dim.columnsX;
    InterpolatePixelValue(idx, in + buffer_in_offset, in_dim.columnsX, out + buffer_out_offset, out_dim.columnsX,
                          pixel_ratio);
}

}  // namespace

namespace alus::rowresample {

void FillLineFrom(float* in_line, size_t in_size, float* out_line, size_t out_size) {
    const auto ratio = GetRatio(in_size, out_size);
    for (size_t i = 0; i < out_size; i++) {
        InterpolatePixelValue(static_cast<int>(i), in_line, static_cast<int>(in_size), out_line,
                              static_cast<int>(out_size), ratio);
    }
}

void ProcessAndTransferHost(const float* input, RasterDimension input_dimensions, float* output,
                            RasterDimension output_dimension) {
    cuda::DeviceBuffer<float> device_in_buf(input_dimensions.columnsX * input_dimensions.rowsY);
    CHECK_CUDA_ERR(
        cudaMemcpy(device_in_buf.Get(), input, device_in_buf.size() * sizeof(float), cudaMemcpyHostToDevice));
    cuda::DeviceBuffer<float> device_out_buf(output_dimension.columnsX * output_dimension.rowsY);
    Process(device_in_buf.Get(), input_dimensions, device_out_buf.Get(), output_dimension);
    CHECK_CUDA_ERR(
        cudaMemcpy(output, device_out_buf.Get(), device_out_buf.size() * sizeof(float), cudaMemcpyDeviceToHost));
}

void Process(const float* input, RasterDimension input_dimension, float* output, RasterDimension output_dimension) {
    const auto ratio = GetRatio(input_dimension.columnsX, output_dimension.columnsX);
    dim3 blockdim(16, 16);
    dim3 griddim((output_dimension.columnsX + blockdim.x - 1) / blockdim.x, (output_dimension.rowsY + blockdim.y - 1) / blockdim.y);
    ResampleKernel<<<griddim, blockdim>>>(input, input_dimension, output, output_dimension, ratio);
    CHECK_CUDA_ERR(cudaDeviceSynchronize());
    CHECK_CUDA_ERR(cudaGetLastError());
}

}  // namespace alus::rowresample