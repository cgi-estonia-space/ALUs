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

#include "doppler_centroid.h"

#include "alus_log.h"
#include "checks.h"
#include "cuda_cleanup.h"

namespace {
constexpr int BLOCK_X_SIZE = 32;
constexpr int BLOCK_Y_SIZE = 32;
constexpr int TOTAL_BLOCK_SIZE = BLOCK_X_SIZE * BLOCK_Y_SIZE;
}  // namespace

__global__ void PhaseDifference(const cufftComplex* data, cufftComplex* result, int x_size, int x_stride, int y_size) {
    __shared__ cufftComplex shared_acc[TOTAL_BLOCK_SIZE];

    const int y = threadIdx.y + blockDim.y * blockIdx.y;
    const int x = threadIdx.x + blockDim.x * blockIdx.x;
    const int shared_idx = threadIdx.x + threadIdx.y * BLOCK_X_SIZE;
    const int result_idx = blockIdx.x + blockIdx.y * gridDim.x;

    // each thread calculates a single phase difference between azimuth lines
    cufftComplex diff = {};
    if (x < x_size && y >= 1 && y < y_size) {
        const int prev_idx = (y - 1) * x_stride + x;
        const int cur_idx = y * x_stride + x;

        cufftComplex cur = data[cur_idx];
        cufftComplex prev = data[prev_idx];
        prev = cuConjf(prev);

        diff = cuCmulf(prev, cur);
    }

    // thread block shared memory reduced to single element
    shared_acc[shared_idx] = diff;
    __syncthreads();
    for (int s = TOTAL_BLOCK_SIZE / 2; s > 0; s /= 2) {
        if (shared_idx < s) {
            cufftComplex t = cuCaddf(shared_acc[shared_idx], shared_acc[shared_idx + s]);
            shared_acc[shared_idx] = t;
        }
        __syncthreads();
    }

    if (shared_idx == 0) {
        result[result_idx] = shared_acc[0];
    }
}

// TODO(priit) replace with thrust::reduce?
__global__ void ReduceComplexSingleBlock(cufftComplex* d_array, int n_elem) {
    __shared__ cufftComplex shared_acc[TOTAL_BLOCK_SIZE];
    const int shared_idx = threadIdx.x;

    cufftComplex r = {};
    for (int idx = threadIdx.x; idx < n_elem; idx += TOTAL_BLOCK_SIZE) {
        if (idx < n_elem) {
            r = cuCaddf(r, d_array[idx]);
        }
    }
    shared_acc[shared_idx] = r;
    __syncthreads();

    for (int s = TOTAL_BLOCK_SIZE / 2; s > 0; s /= 2) {
        if (shared_idx < s) {
            cufftComplex t = cuCaddf(shared_acc[shared_idx], shared_acc[shared_idx + s]);
            shared_acc[shared_idx] = t;
        }
        __syncthreads();
    }

    if (shared_idx == 0) {
        d_array[0] = shared_acc[0];
    }
}

namespace alus::palsar {
void CalculateDopplerCentroid(const DevicePaddedImage& d_img, double prf, double& doppler_centroid) {
    const int azimuth_size = d_img.YSize();
    int range_size = d_img.XSize();

    dim3 block_size(BLOCK_X_SIZE, BLOCK_Y_SIZE);
    dim3 grid_size((range_size + block_size.x - 1) / block_size.x, (azimuth_size + block_size.y - 1) / block_size.y);

    const int total_blocks = grid_size.x * grid_size.y;
    size_t bsize = total_blocks * sizeof(cufftComplex);
    cufftComplex* d_sum;
    CHECK_CUDA_ERR(cudaMalloc(&d_sum, bsize));
    CudaMallocCleanup cleanup(d_sum);
    // Each block calculates a sum of 32x32 phase differences to d_sum, to avoid floating point error accumulation
    PhaseDifference<<<grid_size, block_size>>>(d_img.Data(), d_sum, range_size, d_img.XStride(), azimuth_size);
    // Single thread block reducing d_sum, multistep reduction not needed
    ReduceComplexSingleBlock<<<1, 1024>>>(d_sum, total_blocks);

    // final result is at d_sum[0]
    std::complex<float> h_res;
    static_assert(sizeof(h_res) == sizeof(d_sum[0]));
    CHECK_CUDA_ERR(cudaMemcpy(&h_res, d_sum, sizeof(h_res), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERR(cudaGetLastError());

    const double angle = atan2(h_res.imag(), h_res.real());
    const double fraction = angle / (2 * M_PI);
    doppler_centroid = prf * fraction;
    LOGD << "DC accumulated = " << h_res;
    LOGD << "DC fraction = " << fraction;
}
}  // namespace alus::palsar