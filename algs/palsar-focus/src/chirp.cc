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

#include "chirp.h"

#include <cuda_runtime.h>

#include "alus_log.h"
#include "binary_output.h"
#include "checks.h"
#include "cuda_cleanup.h"
#include "cuda_util.h"
#include "math_utils.h"

namespace {

void ApplyWindowPad(std::vector<std::complex<float>>& chirp_data, size_t padding_size) {
    alus::palsar::ApplyHammingWindow(chirp_data);

    // normalize the chirp

    chirp_data.resize(padding_size);

    // chirp FFT TODO(priit) use a CPU FFT library for simple ones like this
    cufftHandle plan = {};
    alus::palsar::CufftPlanCleanup fft_cleanup(plan);
    CHECK_CUFFT_ERR(cufftPlan1d(&plan, padding_size, CUFFT_C2C, 1));
    cufftComplex* d_chirp = nullptr;
    size_t byte_sz = padding_size * sizeof(cufftComplex);
    CHECK_CUDA_ERR(cudaMalloc(&d_chirp, byte_sz));
    alus::palsar::CudaMallocCleanup mem_cleanup(d_chirp);
    CHECK_CUDA_ERR(cudaMemcpy(d_chirp, chirp_data.data(), byte_sz, cudaMemcpyHostToDevice));
    CHECK_CUFFT_ERR(cufftExecC2C(plan, d_chirp, d_chirp, CUFFT_FORWARD));

    std::vector<std::complex<float>> h_fft(padding_size);
    CHECK_CUDA_ERR(cudaMemcpy(h_fft.data(), d_chirp, byte_sz, cudaMemcpyDeviceToHost));

    // total power in frequency bins, TODO(priit) equivalent in time domain?
    double scale = 0.0;
    for (const auto& el : h_fft) {
        float mag = std::abs(el);
        scale += mag * mag;
    }

    scale /= padding_size;

    // LOGD << "Chrip scaling = " << sqrt(scale);

    // scale time td chirp
    for (auto& el : chirp_data) {
        el /= sqrt(scale);
    }
}
}  // namespace

namespace alus::palsar {

std::vector<std::complex<float>> GenerateChirpData(const ChirpInfo& chirp, size_t padding_size) {
    double Kr = chirp.coefficient[1];

    const int64_t n_samples = chirp.n_samples;
    const double dt = 1.0 / chirp.range_sampling_rate;

    std::vector<std::complex<float>> chirp_data(n_samples);

    for (int64_t i = 0; i < n_samples; i++) {
        const double t = (i - n_samples / 2) * dt;
        const double phase = M_PI * Kr * t * t;
        std::complex<float> iq(cos(phase), sin(phase));

        // iq /= sqrt(n_samples);

        chirp_data[i] = iq;
    }

    ApplyWindowPad(chirp_data, padding_size);

    return chirp_data;
}
}  // namespace alus::palsar