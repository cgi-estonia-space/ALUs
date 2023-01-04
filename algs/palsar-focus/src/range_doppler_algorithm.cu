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

#include "range_doppler_algorithm.h"

#include "alus_log.h"
#include "cuda_cleanup.h"
#include "cufft_plan.h"

#include "checks.h"

#define INPLACE_AZIMUTH_FFT 0  // 0 -> transpose + range FFT + transpose, 1 -> azimuth FFT

// 0 -> Nearest neighbor / no interpolator
//  -> 8 point windowed sinc interpolator
#define RCMC_INTERPOLATOR_SIZE 16

#if defined(RCMC_INTERPOLATOR_SIZE) && RCMC_INTERPOLATOR_SIZE == 0

#elif RCMC_INTERPOLATOR_SIZE == 8

// all windows generated with NumPy 1.17.4

const int N_INTERPOLATOR = RCMC_INTERPOLATOR_SIZE;
// No window
//__constant__ const float WINDOW[8] = {1, 1, 1, 1, 1, 1, 1};

// numpy.blackman(8)
__constant__ const float WINDOW[8] = {-1.38777878e-17, 9.04534244e-02, 4.59182958e-01, 9.20363618e-01,
                                      9.20363618e-01,  4.59182958e-01, 9.04534244e-02, -1.38777878e-17};
// numpy.hamming(8)
//__constant__ const float WINDOW[8] = {0.08,       0.25319469, 0.64235963, 0.95444568,
//                                      0.95444568, 0.64235963, 0.25319469, 0.08};

#elif RCMC_INTERPOLATOR_SIZE == 16
//__constant__ const float WINDOW[16] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
// numpy.blackman(16)
__constant__ const float WINDOW[16] = {-1.38777878e-17, 1.67577197e-02, 7.70724198e-02, 2.00770143e-01,
                                       3.94012424e-01,  6.30000000e-01, 8.49229857e-01, 9.82157437e-01,
                                       9.82157437e-01,  8.49229857e-01, 6.30000000e-01, 3.94012424e-01,
                                       2.00770143e-01,  7.70724198e-02, 1.67577197e-02, -1.38777878e-17};

// numpy.hamming(16)
//__constant__ const float WINDOW[16] = {0.08,       0.11976909, 0.23219992, 0.39785218, 0.58808309, 0.77,
//                                       0.91214782, 0.9899479,  0.9899479,  0.91214782, 0.77,       0.58808309,
//                                       0.39785218, 0.23219992, 0.11976909, 0.08};
#elif RCMC_INTERPOLATOR_SIZE == 32
// numpy.blackman(32)
__constant__ const float WINDOW[32] = {
    -1.38777878e-17, 3.75165430e-03, 1.56384477e-02, 3.74026996e-02, 7.14646070e-02, 1.20286463e-01, 1.85646724e-01,
    2.67954971e-01,  3.65735039e-01, 4.75378537e-01, 5.91228597e-01, 7.06000789e-01, 8.11493284e-01, 8.99490429e-01,
    9.62730703e-01,  9.95797057e-01, 9.95797057e-01, 9.62730703e-01, 8.99490429e-01, 8.11493284e-01, 7.06000789e-01,
    5.91228597e-01,  4.75378537e-01, 3.65735039e-01, 2.67954971e-01, 1.85646724e-01, 1.20286463e-01, 7.14646070e-02,
    3.74026996e-02,  1.56384477e-02, 3.75165430e-03, -1.38777878e-17};
#else
#error "Unsupported RCMC interpolator size"
#endif

#if RCMC_INTERPOLATOR_SIZE > 0
constexpr int N_INTERPOLATOR = RCMC_INTERPOLATOR_SIZE;
static_assert(sizeof(WINDOW) / sizeof(WINDOW[0]) == N_INTERPOLATOR);

__device__ float sinc(float x) {
    const float pi = M_PI;
    if (x == 0) return 1.0f;
    return sinf(pi * x) / (pi * x);
}
#endif

namespace {
struct KernelArgs {
    float Vr;  // TODO(priit) float vs double
    float prf;
    float lambda;
    float doppler_centroid;
    double range_spacing;
    float inv_range_spacing;
    float azimuth_bandwidth_fraction;
    double slant_range;
    int dc_steps;
    bool vv_vh_half_pixel_shift;
};

__global__ void RangeCellMigrationCorrection(const cufftComplex* data_in, int src_x_size, int src_x_stride,
                                             int src_y_size, cufftComplex* data_out, int dst_x_stride,
                                             KernelArgs args) {
    const int dst_x = threadIdx.x + blockIdx.x * blockDim.x;
    const int dst_y = threadIdx.y + blockIdx.y * blockDim.y;

    if (dst_x >= src_x_size || dst_y >= src_y_size) {
        return;
    }

    const float fft_bin_step = args.prf / src_y_size;
    // find FFT bin frequency
    float fn = 0.0f;
    if (dst_y < src_y_size / 2 + args.dc_steps) {
        fn = dst_y * fft_bin_step;
    } else {
        fn = (dst_y - src_y_size) * fft_bin_step;
    }

    // slant range at range pixel
    double R0 = args.slant_range + dst_x * args.range_spacing;

    // Range walk in meters

    double dR = (args.lambda * args.lambda * fn * fn) / (8 * args.Vr * args.Vr);
    dR *= R0;
    // double dR = ((args.lambda * args.lambda) * R0 * fn * fn) / (8 * args.Vr * args.Vr);

    // range walk in pixels
    const float shift = dR / args.range_spacing;
    const int range_walk_pixels = (int)std::round(shift);
    const int dst_idx = dst_x + dst_y * dst_x_stride;

#if RCMC_INTERPOLATOR_SIZE == 0

    if (range_walk_pixels + dst_x < src_x_size) {
        int src_idx = dst_x + range_walk_pixels + dst_y * src_x_stride;
        data_out[dst_idx] = data_in[src_idx];
    } else {
        data_out[dst_idx] = {};
    }
#else
    const int N = N_INTERPOLATOR;
    float i_sum = 0.0f;
    float q_sum = 0.0f;
    float norm_sum = 0.0f;

    const int calc_x = dst_x + range_walk_pixels;
    const int first_idx = calc_x - N / 2 + dst_y * src_x_stride;

    for (int i = 0; i < N; i++) {
        if (calc_x - N / 2 < 0 || calc_x + N / 2 >= src_x_size) {
            continue;
        }

        const cufftComplex data = data_in[first_idx + i];

        const float sinc_offset = shift - range_walk_pixels + N / 2;

        const float mult = WINDOW[i] * sinc(sinc_offset - i);
        i_sum += data.x * mult;
        q_sum += data.y * mult;
        norm_sum += mult;
    }

    if (norm_sum == 0.0f) {
        data_out[dst_idx] = {};
    } else {
        data_out[dst_idx] = {i_sum / norm_sum, q_sum / norm_sum};
    }
#endif
}

__global__ void AzimuthReferenceMultiply(cufftComplex* src_data, int src_width, int src_height, int src_x_stride,
                                         KernelArgs args) {
    int dst_x = threadIdx.x + blockIdx.x * blockDim.x;
    int dst_y = threadIdx.y + blockIdx.y * blockDim.y;

    const float float_pi = M_PI;
    if ((dst_x < src_width && dst_y < src_height)) {
        int src_idx = dst_x + src_x_stride * dst_y;

        const float fft_bin_step = args.prf / src_height;
        // find FFT bin frequency
        float fn = 0.0f;
        float half_pixel_shift = 0.0f;
        if (dst_y < src_height / 2 + args.dc_steps) {
            fn = dst_y * fft_bin_step;
            if (args.vv_vh_half_pixel_shift) {
                half_pixel_shift = -float_pi * static_cast<float>(dst_y) / src_height;
            }
        } else {
            fn = (dst_y - src_height) * fft_bin_step;
            if (args.vv_vh_half_pixel_shift) {
                half_pixel_shift = -float_pi * static_cast<float>(dst_y - src_height) / src_height;
            }
        }

        const float max_freq = (src_height / 2) * fft_bin_step + args.dc_steps * fft_bin_step;
        const float min_freq = -(src_height / 2) * fft_bin_step + args.dc_steps * fft_bin_step;
        if (fn > args.azimuth_bandwidth_fraction * max_freq || fn < args.azimuth_bandwidth_fraction * min_freq) {
            src_data[src_idx] = {};
            return;
        }

        // slant range at range pixel
        const double R0 = args.slant_range + args.range_spacing * dst_x;

        // Azimuth FM rate
        const float Ka = (2 * args.Vr * args.Vr) / (args.lambda * R0);

        // range Doppler domain matched filter at this frequency
        float phase = (-float_pi * fn * fn) / Ka;

        if (args.vv_vh_half_pixel_shift) {
            phase += half_pixel_shift;
        }

        cufftComplex val;
        float sin_val;
        float cos_val;
        sincos(phase, &sin_val, &cos_val);
        val.x = cos_val;
        val.y = sin_val;

        // application via f domain multiplication
        src_data[src_idx] = cuCmulf(src_data[src_idx], val);
    }
}
}  // namespace

#define INPLACE_AZIMUTH_FFT 0

namespace alus::palsar {
void RangeDopplerAlgorithm(const SARMetadata& metadata, palsar::DevicePaddedImage& src_img,
                           palsar::DevicePaddedImage& out_img, CudaWorkspace d_workspace) {
#if INPLACE_AZIMUTH_FFT == 0
    {
        // 2D time domain -> range Doppler domain via Azimuth FFT
        // however, transpose -> range FFT -> transpose is faster than direct Azimuth FFT on all GPUs tested so far
        src_img.Transpose(d_workspace);
        auto azimuth_fft = palsar::PlanRangeFFT(src_img.XStride(), src_img.YSize(), false);
        palsar::CufftPlanCleanup fft_cleanup(azimuth_fft);
        CheckCufftSize(d_workspace.ByteSize(), azimuth_fft);
        CHECK_CUFFT_ERR(cufftSetWorkArea(azimuth_fft, d_workspace.Get()));
        CHECK_CUFFT_ERR(cufftExecC2C(azimuth_fft, src_img.Data(), src_img.Data(), CUFFT_FORWARD));
        // TODO(priit) not actually needed, can fixed by inverting indexing in RCMC and Azimuth Ref multiply
        src_img.Transpose(d_workspace);
    }
#else
    {
        auto azimuth_fft = palsar::PlanAzimuthFFT(src_img.XSize(), src_img.YStride(), src_img.XStride(), false);
        palsar::CufftPlanCleanup fft_cleanup(azimuth_fft);
        CheckCufftSize(d_workspace.ByteSize(), azimuth_fft);
        CHECK_CUFFT_ERR(cufftSetWorkArea(azimuth_fft, d_workspace.Get()));
        CHECK_CUFFT_ERR(cufftExecC2C(azimuth_fft, src_img.Data(), src_img.Data(), CUFFT_FORWARD));
    }
#endif

    double Vr = metadata.results.Vr;
    LOGD << "processing velocity = " << Vr;

    KernelArgs k_args = {};
    k_args.lambda = metadata.wavelength;
    k_args.range_spacing = metadata.range_spacing;
    k_args.inv_range_spacing = 1 / metadata.range_spacing;
    k_args.slant_range = metadata.slant_range_first_sample;
    k_args.Vr = Vr;
    k_args.prf = metadata.pulse_repetition_frequency;
    k_args.doppler_centroid = metadata.results.doppler_centroid;
    k_args.azimuth_bandwidth_fraction = 0.8;
    k_args.dc_steps = metadata.results.doppler_centroid / (metadata.pulse_repetition_frequency / src_img.YStride());
    if (metadata.polarisation == "VV" || metadata.polarisation == "VH") {
        k_args.vv_vh_half_pixel_shift = true;
    }

    LOGD << "Doppler centroid steps = " << k_args.dc_steps;

    // give d_workspace memory to out_img
    out_img.InitExtPtr(src_img.XSize(), src_img.YSize(), src_img.XStride(), src_img.YStride(), d_workspace);

    dim3 block_size(16, 16);
    dim3 grid_size((src_img.XSize() + 15) / 16, (src_img.YStride() + 15) / 16);

    // RCMC step
    RangeCellMigrationCorrection<<<grid_size, block_size>>>(src_img.Data(), src_img.XSize(), src_img.XStride(),
                                                            src_img.YStride(), out_img.Data(), out_img.XStride(),
                                                            k_args);

    // Azimuth Compression step
    AzimuthReferenceMultiply<<<grid_size, block_size>>>(out_img.Data(), out_img.XSize(), out_img.YStride(),
                                                        out_img.XStride(), k_args);

    // src memory no longer needed and can now be used for workspace
    d_workspace = src_img.ReleaseMemory();

    // range Doppler domain back to 2D time domain
#if INPLACE_AZIMUTH_FFT == 0
    {
        // tranpose -> Range IFFT -> transpose instead of direct Azimuth IFFT
        out_img.Transpose(d_workspace);
        auto azimuth_fft = palsar::PlanRangeFFT(out_img.XStride(), out_img.YSize(), false);
        palsar::CufftPlanCleanup fft_cleanup(azimuth_fft);
        CheckCufftSize(d_workspace.ByteSize(), azimuth_fft);
        CHECK_CUFFT_ERR(cufftSetWorkArea(azimuth_fft, d_workspace.Get()));
        CHECK_CUFFT_ERR(cufftExecC2C(azimuth_fft, out_img.Data(), out_img.Data(), CUFFT_INVERSE));
        out_img.Transpose(d_workspace);
    }
#else
    {
        auto azimuth_fft = palsar::PlanAzimuthFFT(out_img.XSize(), out_img.YStride(), out_img.XStride(), false);
        palsar::CufftPlanCleanup fft_cleanup(azimuth_fft);
        CheckCufftSize(d_workspace.ByteSize(), azimuth_fft);
        CHECK_CUFFT_ERR(cufftSetWorkArea(azimuth_fft, d_workspace.Get()));
        CHECK_CUFFT_ERR(cufftExecC2C(azimuth_fft, out_img.Data(), out_img.Data(), CUFFT_INVERSE));
    }
#endif

    // scale FFT -> IFFT roundtrip
    out_img.MultiplyData(1.0f / out_img.YStride());

    CHECK_CUDA_ERR(cudaDeviceSynchronize());
    CHECK_CUDA_ERR(cudaGetLastError());
}
}  // namespace alus::palsar