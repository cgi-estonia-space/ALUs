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

#include "range_compression.h"

#include "checks.h"
#include "cuda_cleanup.h"
#include "cuda_util.h"
#include "cufft_plan.h"
#include "math_utils.h"

__global__ void FrequencyDomainMultiply(cufftComplex* data_fft, const cufftComplex* chirp_fft, int range_fft_size,
                                        int azimuth_size) {
    const int x = threadIdx.x + blockDim.x * blockIdx.x;
    const int y = threadIdx.y + blockDim.y * blockIdx.y;
    const int data_idx = y * range_fft_size + x;

    if (x < range_fft_size && y < azimuth_size) {
        cufftComplex chirp_bin = chirp_fft[x];

        // Conjugate chirp FFT bin before multiplication
        chirp_bin = cuConjf(chirp_bin);

        cufftComplex data_bin = data_fft[data_idx];

        data_fft[data_idx] = cuCmulf(data_bin, chirp_bin);
    }
}

__global__ void FinishRangeCompression(cufftComplex* data_fft, int range_size, int cutoff, int azimuth_size,
                                       float multiplier) {
    const int x = threadIdx.x + blockDim.x * blockIdx.x;
    const int y = threadIdx.y + blockDim.y * blockIdx.y;
    const int data_idx = y * range_size + x;

    if (x < cutoff && y < azimuth_size) {
        data_fft[data_idx].x *= multiplier;
        data_fft[data_idx].y *= multiplier;
    } else if (x < range_size && y < azimuth_size) {
        data_fft[data_idx] = {};
    }
}

namespace alus::palsar {

void RangeCompression(DevicePaddedImage& data, const std::vector<std::complex<float>>& chirp_data, int chirp_samples,
                      CudaWorkspace& d_workspace) {
    const int range_fft_size = data.XStride();
    const int azimuth_size = data.YSize();
    const int chirp_size = static_cast<int>(chirp_data.size());

    if (range_fft_size != chirp_size) {
        throw std::logic_error("Range padding and chirp size mismatch");
    }

    // chirp host -> device
    cufftComplex* d_chirp;
    static_assert(sizeof(cufftComplex) == sizeof(chirp_data[0]));

    const size_t chirp_bsize = 8 * chirp_data.size();
    CHECK_CUDA_ERR(cudaMalloc(&d_chirp, chirp_bsize));
    palsar::CudaMallocCleanup chirp_cleanup(d_chirp);
    CHECK_CUDA_ERR(cudaMemcpy(d_chirp, chirp_data.data(), chirp_bsize, cudaMemcpyHostToDevice));

    {
        // inplace FFT of chirp
        // NB! conjugation applied during frequency domain multiplication
        // TODO(priit) do this on CPU?
        cufftHandle chirp_plan;
        CHECK_CUFFT_ERR(cufftPlan1d(&chirp_plan, range_fft_size, CUFFT_C2C, 1));
        palsar::CufftPlanCleanup plan_cleanup(chirp_plan);
        CHECK_CUFFT_ERR(cufftExecC2C(chirp_plan, d_chirp, d_chirp, CUFFT_FORWARD));
    }

    cuComplex* d_data = data.Data();
    // inplace FFT on each range row of the SAR data
    cufftHandle range_fft_plan = PlanRangeFFT(range_fft_size, azimuth_size, false);
    palsar::CufftPlanCleanup range_fft_cleanup(range_fft_plan);
    CheckCufftSize(d_workspace.ByteSize(), range_fft_plan);
    CHECK_CUFFT_ERR(cufftSetWorkArea(range_fft_plan, d_workspace.Get()));
    CHECK_CUFFT_ERR(cufftExecC2C(range_fft_plan, d_data, d_data, CUFFT_FORWARD));

    // matched filter via frequency domain multiplication
    dim3 block_size(16, 16);
    dim3 grid_size((range_fft_size + 15) / 16, (azimuth_size + 15) / 16);
    FrequencyDomainMultiply<<<grid_size, block_size>>>(d_data, d_chirp, range_fft_size, azimuth_size);

    // Back to time domain
    CHECK_CUFFT_ERR(cufftExecC2C(range_fft_plan, d_data, d_data, CUFFT_INVERSE));

    const int new_range_size = data.XSize() - chirp_samples;
    data.SetXSize(new_range_size);

    // scale FFT -> IFFT roundtrip and zero memory of the padding
    FinishRangeCompression<<<grid_size, block_size>>>(d_data, range_fft_size, new_range_size, azimuth_size,
                                                      1.0 / range_fft_size);

    CHECK_CUDA_ERR(cudaDeviceSynchronize());  // not needed at this point, but helps with debugging
    CHECK_CUDA_ERR(cudaGetLastError());
}
}  // namespace alus::palsar
