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

#include "dc_bias.h"

#include "cuda_cleanup.h"

namespace {
using alus::palsar::IQ8;

constexpr uint8_t NO_DATA = 0xFF; // no data due to differing slant ranges between rows

__global__ void CorrectRawKernel(const IQ8* data_in, IQ8* data_out, int x_size, int y_size, int row_offset,
                                 uint32_t* offsets) {
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;

    const uint32_t offset = offsets[y];

    const int in_idx = x + row_offset + (y * (x_size + row_offset));

    if (x < x_size && y < y_size) {
        const int out_idx = x + y * x_size;
        if (offset == 0) {
            data_out[out_idx] = data_in[in_idx];
        } else if (static_cast<uint32_t>(x) < offset) {
            data_out[out_idx] = {NO_DATA, NO_DATA};
        } else if (offset > 0 && ((static_cast<int>(offset) + x) < x_size)) {
            data_out[out_idx + offset] = data_in[in_idx];
        }
    }
}

__global__ void ReduceSumRawIQData(const IQ8* data, int x_size, int y_size, unsigned long long* result) {
    const int y_start = blockIdx.y;
    const int x_start = threadIdx.x;
    const int y_step = gridDim.y;
    const int x_step = blockDim.x;

    uint32_t n_samples = 0;
    uint32_t acc_i = 0;
    uint32_t acc_q = 0;
    for (int y = y_start; y < y_size; y += y_step) {
        for (int x = x_start; x < x_size; x += x_step) {
            int idx = (y * x_size) + x;
            auto sample = data[idx];
            if (sample.i == NO_DATA && sample.q == NO_DATA) {
                continue;
            }

            acc_i += sample.i;
            acc_q += sample.q;
            n_samples++;
        }
    }

    // TODO shared reduction before atomics, if this should ever matter
    //  3 x SM count x 1024 atomicAdds on one memory location should not be too bad
    atomicAdd(result, acc_i);
    atomicAdd(result + 1, acc_q);
    atomicAdd(result + 2, n_samples);
}

__global__ void ApplyDCBiasKernel(const IQ8* src, int src_x_size, cufftComplex* dest, int dest_x_size,
                                  int dest_x_stride, int dest_y_size, int dest_y_stride, float dc_i, float dc_q) {
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;

    const int dst_idx = y * dest_x_stride + x;
    const int src_idx = y * src_x_size + x;
    if (x < dest_x_size && y < dest_y_size) {
        // signal data index, convert from 2 x uint8_t to 2 x Float32
        auto iq8 = src[src_idx];
        if (iq8.i == NO_DATA && iq8.q == NO_DATA) {
            dest[dst_idx] = {};
        } else {
            dest[dst_idx].x = static_cast<float>(iq8.i) - dc_i;
            dest[dst_idx].y = static_cast<float>(iq8.q) - dc_q;
        }

    } else if (x < dest_x_stride && y < dest_y_stride) {
        // zero pad index, both in range and azimuth
        dest[dst_idx] = {0.0f, 0.0f};
    }
}
}  // namespace

namespace alus::palsar {

void FormatRawIQ8(const IQ8* d_signal_in, IQ8* d_signal_out, ImgFormat img, const std::vector<uint32_t>& offsets) {
    uint32_t* d_offsets;
    const size_t byte_size = offsets.size() * sizeof(offsets[0]);
    CHECK_CUDA_ERR(cudaMalloc(&d_offsets, byte_size));
    CudaMallocCleanup cleanup(d_offsets);
    CHECK_CUDA_ERR(cudaMemcpy(d_offsets, offsets.data(), byte_size, cudaMemcpyHostToDevice));
    dim3 block_size(16, 16);
    dim3 grid_size((img.range_size + 15) / 16, (img.azimuth_size + 15) / 16);

    static_assert(sizeof(IQ8) == 2);
    const int row_offset = img.data_line_offset / 2;  // accessed via IQ8*

    CorrectRawKernel<<<grid_size, block_size>>>(d_signal_in, d_signal_out, img.range_size, img.azimuth_size, row_offset,
                                                d_offsets);

    CHECK_CUDA_ERR(cudaDeviceSynchronize());
    CHECK_CUDA_ERR(cudaGetLastError());
}

void CalculateDCBias(const IQ8* d_signal_data, ImgFormat img, size_t multiprocessor_count, SARResults& result) {
    unsigned long long* d_result;
    const size_t res_bsize = 8 * 3;
    CHECK_CUDA_ERR(cudaMalloc(&d_result, res_bsize));
    CudaMallocCleanup cleanup(d_result);
    CHECK_CUDA_ERR(cudaMemset(d_result, 0, res_bsize));
    dim3 block_size(1024);
    dim3 grid_size(1, multiprocessor_count * 2);

    // Accumulate I and Q over the whole dataset
    ReduceSumRawIQData<<<grid_size, block_size>>>(d_signal_data, img.range_size, img.azimuth_size, d_result);
    unsigned long long h_result[3];
    CHECK_CUDA_ERR(cudaMemcpy(h_result, d_result, res_bsize, cudaMemcpyDeviceToHost));

    const int total_samples = h_result[2];

    // find the average signal value for I and Q, should be ~15.5f for 5 bit PALSAR data;
    result.dc_i = static_cast<double>(h_result[0]) / total_samples;
    result.dc_q = static_cast<double>(h_result[1]) / total_samples;
    result.total_samples = total_samples;
}

void ApplyDCBias(const IQ8* d_signal_data, const SARResults& results, ImgFormat img, DevicePaddedImage& output) {
    dim3 block_size(32, 32);
    dim3 grid_size((output.XStride() - 31) / 32, (output.YStride() - 31) / 32);
    ApplyDCBiasKernel<<<grid_size, block_size>>>(d_signal_data, img.range_size, output.Data(),
                                                 output.XSize(), output.XStride(), output.YSize(), output.YStride(),
                                                 results.dc_i, results.dc_q);

    CHECK_CUDA_ERR(cudaDeviceSynchronize());
    CHECK_CUDA_ERR(cudaGetLastError());
}
}  // namespace alus::palsar