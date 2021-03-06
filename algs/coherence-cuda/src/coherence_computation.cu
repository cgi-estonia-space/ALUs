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

#include <tuple>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include <cublas_v2.h>
#include <cusolverDn.h>

#include <thrust/complex.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/functional.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>

#include "band_params.h"
#include "coh_window.h"
#include "coherence_computation.h"
#include "helper_cuda.h"

namespace {
struct Power {
    __host__ __device__ double operator()(const double x, const double power) { return pow(x, power); }
};

struct GetA {
    __host__ __device__ thrust::tuple<double> operator()(const thrust::tuple<double, double, double, double>& t) {
        //        todo: check if we can use shorts as input here
        double x = thrust::get<0>(t);
        double x_pow = thrust::get<1>(t);
        double y = thrust::get<2>(t);
        double y_pow = thrust::get<3>(t);
        return thrust::make_tuple(pow(x, x_pow) * pow(y, y_pow));
    }
};

struct NormalizeDouble {
    static constexpr double hf = 0.5;
    static constexpr double qt = 0.25;
    const double min;
    const double max;
    NormalizeDouble(double min, double max) : min(min), max(max) {}
    __host__ __device__ double operator()(const double& x) { return (x - hf * (min + max)) / (qt * (max - min)); }
};

struct DataMasterNorm {
    __host__ __device__ thrust::tuple<float, float> operator()(const thrust::tuple<float, float, float, float>& t) {
        float master_real = thrust::get<0>(t);
        float master_imaginary = thrust::get<1>(t);
        float slave_real = thrust::get<2>(t);
        float slave_imaginary = thrust::get<3>(t);
        auto out = thrust::complex<float>(master_real, master_imaginary) *
                   thrust::conj(thrust::complex<float>(slave_real, slave_imaginary));
        return thrust::make_tuple(out.real(), out.imag());
    }
};

struct Norm {
    __host__ __device__ thrust::tuple<float> operator()(const thrust::tuple<float, float>& t) {
        float real = thrust::get<0>(t);
        float imaginary = thrust::get<1>(t);
        auto out = (real * real) + (imaginary * imaginary);
        return thrust::make_tuple(out);
    }
};

struct SlaveMultiplyComplexReferencePhase {
    __host__ __device__ thrust::tuple<float, float> operator()(const thrust::tuple<double, float, float>& t) {
        auto flat_earth_phase = thrust::get<0>(t);
        float slave_real = thrust::get<1>(t);
        float slave_imaginary = thrust::get<2>(t);
        auto out = thrust::complex<float>(slave_real, slave_imaginary) *
                   thrust::complex<float>(static_cast<float>(cos(flat_earth_phase)),
                                          static_cast<float>(sin(flat_earth_phase)));
        return thrust::make_tuple(out.real(), out.imag());
    }
};

struct FilteredCoherenceProduct {
    __host__ __device__ thrust::tuple<float> operator()(const thrust::tuple<float, float, float, float, bool>& t) {
        float master_real = thrust::get<0>(t);
        float master_imaginary = thrust::get<1>(t);
        float slave_real = thrust::get<2>(t);
        float slave_imaginary = thrust::get<3>(t);
        bool keep_pixel = thrust::get<4>(t);
        auto product_t = slave_real * slave_imaginary;
        if (keep_pixel && product_t > 0) {
            return thrust::make_tuple(thrust::abs(thrust::complex<float>(master_real, master_imaginary)) /
                                      sqrt(product_t));
        }
        return thrust::make_tuple(0);
    }
};
}  // namespace

namespace alus {
namespace coherence_cuda {

/**
 * input tile contains overlap data but no padding
 * output tile is smaller since overlaps get removed
 */
__global__ void SimpleCoherence2DSumKernelSumSurroundings(float* d_tile_in_data_ptr, float* d_tile_out_data_ptr,
                                                          int input_tile_width, int input_tile_height,
                                                          int output_tile_width, int output_tile_height,
                                                          int coh_window_rg, int coh_window_az, int x_min_pad,
                                                          int x_max_pad, int y_min_pad, int y_max_pad) {
    int column_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int row_idx = blockIdx.y * blockDim.y + threadIdx.y;

    int column_idx_input_tile = column_idx - x_min_pad;
    int row_idx_input_tile = row_idx - y_min_pad;

    if (row_idx < output_tile_height && column_idx < output_tile_width) {
        auto coh_window_rg_size = coh_window_rg;
        auto start_range_x = 0;
        if (column_idx_input_tile < 0) {
            start_range_x = abs(column_idx_input_tile);
        }

        if (column_idx + x_max_pad > input_tile_width - 1) {
            coh_window_rg_size = coh_window_rg + (input_tile_width - 1 - (column_idx + x_max_pad));
        }

        auto coh_window_az_size = coh_window_az;
        auto start_range_y = 0;
        if (row_idx_input_tile < 0) {
            start_range_y = abs(row_idx_input_tile);
        }

        if (row_idx + y_max_pad > input_tile_height - 1) {
            coh_window_az_size = coh_window_az + (input_tile_height - 1 - (row_idx + y_max_pad));
        }

        float pixel_value = 0.0f;
        for (int i = start_range_y; i < coh_window_az_size; i++) {
            float acc = 0.0f;
            for (int j = start_range_x; j < coh_window_rg_size; j++) {
                const int idx = (row_idx_input_tile + i) * input_tile_width + (column_idx_input_tile + j);
                if (idx < (input_tile_width * input_tile_height)) {
                    acc += d_tile_in_data_ptr[idx];
                }
            }
            pixel_value += acc;
        }

        d_tile_out_data_ptr[row_idx * output_tile_width + column_idx] = pixel_value;
    }
}

/**
 * input tile contains overlap data but no padding
 * output tile is smaller since overlaps get removed
 */
__global__ void BoolImageForCoherenceProductFiltering(float* d_tile_in_data_ptr, bool* d_tile_out_data_ptr,
                                                      int input_tile_width, int input_tile_height,
                                                      int output_tile_width, int output_tile_height, int coh_window_rg,
                                                      int coh_window_az, int x_min_pad, int x_max_pad, int y_min_pad,
                                                      int y_max_pad) {
    const float slave_real_no_data = 0.0F;
    auto min_cut_azw = (coh_window_az - 1) / 2;
    auto min_cut_rgw = (coh_window_rg - 1) / 2;

    int column_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int row_idx = blockIdx.y * blockDim.y + threadIdx.y;

    // edge tiles start from same index, other tiles input starts half coherence window before
    int column_idx_input_tile = column_idx + min_cut_rgw - x_min_pad;
    int row_idx_input_tile = row_idx + min_cut_azw - y_min_pad;

    if (row_idx < output_tile_height && column_idx < output_tile_width) {
        auto data_in = d_tile_in_data_ptr[row_idx_input_tile * input_tile_width + column_idx_input_tile];
        d_tile_out_data_ptr[row_idx * output_tile_width + column_idx] =
            std::fabs(data_in - slave_real_no_data) >= std::numeric_limits<float>::epsilon();
    }
}

void CoherenceComputation::Linspance(double min, double max, cuda::DeviceBuffer<double>& d_vector,
                                     cudaStream_t stream) {
    double delta = (max - min) / static_cast<double>(d_vector.GetElemCount() - 1);
    thrust::transform(thrust::cuda::par.on(stream), thrust::make_counting_iterator(min / delta),
                      thrust::make_counting_iterator((max + 1.) / delta), thrust::make_constant_iterator(delta),
                      d_vector.begin(), thrust::multiplies<double>());
}

void CoherenceComputation::MatMulAB(cublasHandle_t handle, const double* A, const double* B, double* C, const int m,
                                    const int k, const int n) {
    const double alf = 1;
    const double bet = 0;
    const double* alpha = &alf;
    const double* beta = &bet;

    CHECK_CUDA_ERRORS(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, A, m, B, k, beta, C, m));
}

void CoherenceComputation::MatMulATransposeB(cublasHandle_t handle, const double* A, const double* B, double* C,
                                             const int m, const int k, const int n) {
    const double alf = 1;
    const double bet = 0;
    const double* alpha = &alf;
    const double* beta = &bet;
    CHECK_CUDA_ERRORS(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, m, n, k, alpha, A, m, B, n, beta, C, m));
}

void CoherenceComputation::MatMulATransposeA(cublasHandle_t handle, const double* A, double* C, const int m,
                                             const int n) {
    const double alf = 1;
    const double bet = 0;
    const double* alpha = &alf;
    const double* beta = &bet;
    CHECK_CUDA_ERRORS(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, m, m, n, alpha, A, m, A, m, beta, C, m));
}

void CoherenceComputation::TransposeA(cublasHandle_t handle, const double* A, double* C, const int m, const int n) {
    const double alf = 1;
    const double bet = 0;
    const double* alpha = &alf;
    const double* beta = &bet;
    CHECK_CUDA_ERRORS(cublasDgeam(handle, CUBLAS_OP_T, CUBLAS_OP_T, m, n, alpha, A, n, beta, A, n, C, m));
}

void CoherenceComputation::GenerateA(cublasHandle_t handle, thrust::device_vector<double>& d_x,
                                     thrust::device_vector<double>& d_y, thrust::device_vector<double>& d_out,
                                     double x_norm_min, double x_norm_max, double y_norm_min, double y_norm_max) {
    // normalize vectors
    thrust::transform(d_x.begin(), d_x.end(), d_x.begin(), NormalizeDouble(x_norm_min, x_norm_max));
    thrust::transform(d_y.begin(), d_y.end(), d_y.begin(), NormalizeDouble(y_norm_min, y_norm_max));
    thrust::device_vector<double> d_x_xpows_ones_t(d_ones_.size() * d_x.size());

    // pixels to correct shape using outer product of two vectors
    // reusing d_out for y side
    MatMulATransposeB(handle, thrust::raw_pointer_cast(d_x.data()), thrust::raw_pointer_cast(d_ones_.data()),
                      thrust::raw_pointer_cast(d_out.data()), d_x.size(), 1, d_ones_.size());

    MatMulATransposeB(handle, thrust::raw_pointer_cast(d_y.data()), thrust::raw_pointer_cast(d_ones_.data()),
                      thrust::raw_pointer_cast(d_x_xpows_ones_t.data()), d_y.size(), 1, d_ones_.size());

    //    todo: check if dimension matching would be better when we use ones with less precision
    thrust::device_vector<double> d_y_ones_ypows_t(d_y_pows_.size() * d_x.size());
    thrust::device_vector<double> d_x_ones_xpows_t(d_x_pows_.size() * d_y.size());
    // since x and y are same size we can use single ones vector
    thrust::device_vector<double> d_xy_ones(d_x.size());
    thrust::fill(d_xy_ones.begin(), d_xy_ones.end(), 1.0);

    // x and y powers to correct shape using outer product of two vectors,  k = 1 because we have vectors
    MatMulATransposeB(handle, thrust::raw_pointer_cast(d_xy_ones.data()), thrust::raw_pointer_cast(d_y_pows_.data()),
                      thrust::raw_pointer_cast(d_y_ones_ypows_t.data()), d_xy_ones.size(), 1, d_y_pows_.size());
    MatMulATransposeB(handle, thrust::raw_pointer_cast(d_xy_ones.data()), thrust::raw_pointer_cast(d_x_pows_.data()),
                      thrust::raw_pointer_cast(d_x_ones_xpows_t.data()), d_xy_ones.size(), 1, d_x_pows_.size());

    //    todo: check if single output needs to make tuple here
    // element-wise powers for x and y and multiply these to get A
    thrust::transform(thrust::make_zip_iterator(thrust::make_tuple(d_x_xpows_ones_t.begin(), d_x_ones_xpows_t.begin(),
                                                                   d_out.begin(), d_y_ones_ypows_t.begin())),
                      thrust::make_zip_iterator(thrust::make_tuple(d_x_xpows_ones_t.end(), d_x_ones_xpows_t.end(),
                                                                   d_out.end(), d_y_ones_ypows_t.end())),
                      thrust::make_zip_iterator(thrust::make_tuple(d_out.begin())), GetA());
}

void CoherenceComputation::LaunchCoherencePreTileCalc(
    std::vector<int>& x_pows, std::vector<int>& y_pows,
    std::tuple<std::vector<int>, std::vector<int>>& position_lines_pixels, std::vector<double>& generate_y,
    const BandParams& band_params) {
    d_x_pows_ = x_pows;
    d_y_pows_ = y_pows;
    thrust::device_vector<double> d_lines = std::get<0>(position_lines_pixels);
    thrust::device_vector<double> d_pixels = std::get<1>(position_lines_pixels);
    thrust::device_vector<double> d_generate_y_vector = generate_y;
    // device vector to be re-used for size matching
    d_ones_ = thrust::device_vector<double>(x_pows.size(), 1.0);
    d_ones_size_ = d_ones_.size();
    subtract_flat_earth_ = true;

    cublasHandle_t handle;
    CHECK_CUDA_ERRORS(cublasCreate(&handle));  // NB! multiple threads should not share the same CUBLAS handle

    //    todo: measure times between here and without refactored to see if something changed
    thrust::device_vector<double> d_a(d_ones_.size() * d_lines.size());

    GenerateA(handle, d_pixels, d_lines, d_a, band_params.band_x_min, band_params.band_x_size - 1,
              band_params.band_y_min, band_params.band_y_size - 1);
    // todo: use transform iterators where possible to avoid storing into memory

    // now calcualte coeficents

    // input 21*501  -> result is 21*21
    thrust::device_vector<double> d_n(d_ones_.size() * d_ones_.size());

    auto a = thrust::raw_pointer_cast(d_a.data());
    auto n = thrust::raw_pointer_cast(d_n.data());
    thrust::device_vector<double> transposed_a(d_a.size());

    TransposeA(handle, a, thrust::raw_pointer_cast(transposed_a.data()), d_ones_.size(), d_pixels.size());
    MatMulATransposeA(handle, thrust::raw_pointer_cast(transposed_a.data()), n, d_ones_.size(), d_pixels.size());

    auto y = thrust::raw_pointer_cast(d_generate_y_vector.data());
    thrust::device_vector<double> d_rhs(x_pows.size());
    auto rhs = thrust::raw_pointer_cast(d_rhs.data());
    MatMulAB(handle, thrust::raw_pointer_cast(transposed_a.data()), y, rhs, d_ones_.size(), d_pixels.size(), 1);

    cusolverDnHandle_t solver_handle;
    CHECK_CUDA_ERRORS(cusolverDnCreate(&solver_handle));

    d_coefs_.resize(d_ones_.size());

    auto coefs = thrust::raw_pointer_cast(d_coefs_.data());  // duplicate size matrix from rhs
    size_t lwork_bytes;

    thrust::device_vector<int> d_info(1);
    auto info = thrust::raw_pointer_cast(d_info.data());
    int niters;
    thrust::device_vector<int> d_dipiv(x_pows.size());  // rhs rows
    auto dipiv = thrust::raw_pointer_cast(d_dipiv.data());

    CHECK_CUDA_ERRORS(cusolverDnDDgesv_bufferSize(solver_handle, d_ones_.size(), 1, n, d_ones_.size(), dipiv, rhs,
                                                  d_ones_.size(), coefs, d_ones_.size(), nullptr, &lwork_bytes));

    double* d_workspace = nullptr;

    //    todo: double check if needs multiplication with sizeof(double) here  at start I did not see the need, but
    //    examples is hard to understand, still not sure if this needs to be multiplied by sizeof(double), seems like
    //    works without, but this might be luck...
    CHECK_CUDA_ERRORS(cudaMalloc((void**)&d_workspace, lwork_bytes /** sizeof(double)*/));

    CHECK_CUDA_ERRORS(cusolverDnDDgesv(solver_handle, d_ones_.size(), 1, n, d_ones_.size(), dipiv, rhs, d_ones_.size(),
                                       coefs, d_ones_.size(), d_workspace, lwork_bytes, &niters, info));
    // todo: make optional logging for extra info available?
    //    std::cout << "niters: " << niters << std::endl;
    //    std::cout << "info: " << info << std::endl;

    if (d_workspace) {
        CHECK_CUDA_ERRORS(cudaFree(d_workspace));
    }
    if (solver_handle) {
        CHECK_CUDA_ERRORS(cusolverDnDestroy(solver_handle));
    }
    if (handle) {
        CHECK_CUDA_ERRORS(cublasDestroy(handle));
    }
    //    if(true){
    //        cudaDeviceReset();
    //    }
}

void CoherenceComputation::LaunchCoherence(const CohTile& tile, ThreadContext& ctx, const CohWindow& coh_window,
                                           const BandParams& band_params) {
    const int input_tile_width = tile.GetTileIn().GetXSize();
    const int input_tile_height = tile.GetTileIn().GetYSize();
    const int output_tile_width = tile.GetTileOut().GetXSize();
    const int output_tile_height = tile.GetTileOut().GetYSize();

    auto thrust_stream = thrust::cuda::par.on(ctx.stream);
    // how threads are in block
    //    dim3 threads_per_block(32, 32, 1);  // this should be 32 <= x*y*z <=1024 & warp size based (multiple of 32)
    //    int output_size = output_tile_width * output_tile_height;

    // thread block limits: x_max:1024, y_max:1024 and z_max:64 && x ?? y ?? z ??? 1024
    // x_width_max * y_limit_max = 1024
    dim3 threads_per_block(32, 32, 1);
    dim3 num_blocks((output_tile_width + threads_per_block.x - 1) / threads_per_block.x,
                    (output_tile_height + threads_per_block.y - 1) / threads_per_block.y);

    ctx.d_tile_out_slave_real_bool.Resize(output_tile_width * output_tile_height);

    BoolImageForCoherenceProductFiltering<<<num_blocks, threads_per_block, 0, ctx.stream>>>(
        ctx.d_band_slave_real.Get(), ctx.d_tile_out_slave_real_bool.Get(), input_tile_width, input_tile_height,
        output_tile_width, output_tile_height, coh_window.rg, coh_window.az, tile.GetXMinPad(), tile.GetXMaxPad(),
        tile.GetYMinPad(), tile.GetYMaxPad());
    CHECK_CUDA_ERRORS(cudaPeekAtLastError());

    size_t size_to_last = input_tile_height * input_tile_width;
    // ComputeFlatEarthPhase
    if (subtract_flat_earth_) {
        ctx.d_range_axis.Resize(input_tile_width);
        ctx.d_azimuth_axis.Resize(input_tile_height);
        const auto& tile_in = tile.GetTileIn();
        Linspance(tile_in.GetXMin(), tile_in.GetXMax(), ctx.d_range_axis, ctx.stream);
        Linspance(tile_in.GetYMin(), tile_in.GetYMax(), ctx.d_azimuth_axis, ctx.stream);

        thrust::transform(thrust_stream, ctx.d_range_axis.begin(), ctx.d_range_axis.end(), ctx.d_range_axis.begin(),
                          NormalizeDouble(band_params.band_x_min, band_params.band_x_size - 1));
        thrust::transform(thrust_stream, ctx.d_azimuth_axis.begin(), ctx.d_azimuth_axis.end(),
                          ctx.d_azimuth_axis.begin(),
                          NormalizeDouble(band_params.band_y_min, band_params.band_y_size - 1));

        cublasHandle_t handle = ctx.handle;

        ctx.d_x_xpows_ones_t.Resize(d_ones_.size() * ctx.d_range_axis.size());
        ctx.d_y_ypows_ones_t.Resize(d_ones_.size() * ctx.d_azimuth_axis.size());

        size_t d_azimuth_axis_size = ctx.d_azimuth_axis.size();
        size_t d_range_axis_size = ctx.d_range_axis.size();
        // 2675*21
        MatMulATransposeB(handle, ctx.d_range_axis.data(), thrust::raw_pointer_cast(d_ones_.data()),
                          ctx.d_x_xpows_ones_t.data(), d_range_axis_size, 1, d_ones_.size());
        // 1503*21
        MatMulATransposeB(handle, ctx.d_azimuth_axis.data(), thrust::raw_pointer_cast(d_ones_.data()),
                          ctx.d_y_ypows_ones_t.data(), d_azimuth_axis_size, 1, d_ones_.size());

        ctx.d_y_ones_ypows_t.Resize(d_ones_size_ * d_azimuth_axis_size);
        ctx.d_x_ones_xpows_t.Resize(d_ones_size_ * d_range_axis_size);

        // SWAPED POWS HERE ON PURPOSE
        ctx.d_y_ones.Resize(d_azimuth_axis_size);
        ctx.d_x_ones.Resize(d_range_axis_size);
        thrust::fill(thrust_stream, ctx.d_y_ones.begin(), ctx.d_y_ones.end(), 1.0);
        thrust::fill(thrust_stream, ctx.d_x_ones.begin(), ctx.d_x_ones.end(), 1.0);

        // 2675*21
        MatMulATransposeB(handle, ctx.d_x_ones.data(), thrust::raw_pointer_cast(d_y_pows_.data()),
                          ctx.d_x_ones_xpows_t.data(), ctx.d_x_ones.size(), 1, d_y_pows_.size());

        // 1503*21
        MatMulATransposeB(handle, ctx.d_y_ones.data(), thrust::raw_pointer_cast(d_x_pows_.data()),
                          ctx.d_y_ones_ypows_t.data(), ctx.d_y_ones.size(), 1, d_x_pows_.size());

        // POWER X AND Y SIDES
        thrust::transform(thrust_stream, ctx.d_x_xpows_ones_t.begin(), ctx.d_x_xpows_ones_t.end(),
                          ctx.d_x_ones_xpows_t.begin(), ctx.d_x_xpows_ones_t.begin(), Power());
        thrust::transform(thrust_stream, ctx.d_y_ypows_ones_t.begin(), ctx.d_y_ypows_ones_t.end(),
                          ctx.d_y_ones_ypows_t.begin(), ctx.d_y_ypows_ones_t.begin(), Power());

        ctx.d_y_ones_coefs_t.Resize(d_coefs_.size() * ctx.d_y_ones.size());

        // Y SIDE COEFS MULTIPLY Y_SIDE_POWERS
        // make coefs same shape
        MatMulATransposeB(handle, ctx.d_y_ones.data(), thrust::raw_pointer_cast(d_coefs_.data()),
                          ctx.d_y_ones_coefs_t.data(), ctx.d_y_ones.size(), 1, d_coefs_.size());

        // multiply y side by coefs
        thrust::transform(thrust_stream, ctx.d_y_ypows_ones_t.begin(), ctx.d_y_ypows_ones_t.end(),
                          ctx.d_y_ones_coefs_t.begin(), ctx.d_y_ypows_ones_t.begin(), thrust::multiplies<double>());

        ctx.d_flat_earth_phase.Resize(d_range_axis_size * d_azimuth_axis_size);

        MatMulATransposeB(handle, ctx.d_x_xpows_ones_t.data(), ctx.d_y_ypows_ones_t.data(),
                          ctx.d_flat_earth_phase.data(), d_range_axis_size, d_ones_size_, d_azimuth_axis_size);

        thrust::transform(
            thrust_stream,
            thrust::make_zip_iterator(thrust::make_tuple(ctx.d_flat_earth_phase.begin(), ctx.d_band_slave_real.begin(),
                                                         ctx.d_band_slave_imag.begin())),
            thrust::make_zip_iterator(thrust::make_tuple(ctx.d_flat_earth_phase.end(),
                                                         ctx.d_band_slave_real.end() /*+ size_to_last*/,
                                                         ctx.d_band_slave_imag.end())),
            thrust::make_zip_iterator(thrust::make_tuple(ctx.d_band_slave_real.begin(), ctx.d_band_slave_imag.begin())),
            SlaveMultiplyComplexReferencePhase());
    }

    ctx.complex_data_slave_norm_imaginary.Resize(input_tile_height * input_tile_width);

    thrust::transform(
        thrust_stream,
        thrust::make_zip_iterator(thrust::make_tuple(ctx.d_band_master_real.begin(), ctx.d_band_master_imag.begin())),
        thrust::make_zip_iterator(thrust::make_tuple(ctx.d_band_master_real.end(), ctx.d_band_master_imag.end())),
        thrust::make_zip_iterator(thrust::make_tuple(ctx.complex_data_slave_norm_imaginary.begin())), Norm());

    ctx.complex_data_slave_norm_real.Resize(input_tile_height * input_tile_width);

    thrust::transform(
        thrust_stream,
        thrust::make_zip_iterator(thrust::make_tuple(ctx.d_band_slave_real.begin(), ctx.d_band_slave_imag.begin())),
        thrust::make_zip_iterator(thrust::make_tuple(ctx.d_band_slave_real.end(), ctx.d_band_slave_imag.end())),
        thrust::make_zip_iterator(thrust::make_tuple(ctx.complex_data_slave_norm_real.begin())), Norm());

    // input is master_real,master_imaginary,slave_real,slave_imaginary
    // output is data_master_norm (check coherence_calc)
    ctx.data_master_norm_real.Resize(size_to_last);
    ctx.data_master_norm_imaginary.Resize(size_to_last);

    thrust::transform(
        thrust_stream,
        thrust::make_zip_iterator(thrust::make_tuple(ctx.d_band_master_real.begin(), ctx.d_band_master_imag.begin(),
                                                     ctx.d_band_slave_real.begin(), ctx.d_band_slave_imag.begin())),
        thrust::make_zip_iterator(thrust::make_tuple(ctx.d_band_master_real.end(), ctx.d_band_master_imag.end(),
                                                     ctx.d_band_slave_real.end(), ctx.d_band_slave_imag.end())),
        thrust::make_zip_iterator(
            thrust::make_tuple(ctx.data_master_norm_real.begin(), ctx.data_master_norm_imaginary.begin())),
        DataMasterNorm());

    // todo: use streams to make custom kernel per layer device level parallel

    ctx.d_tile_out_master_real.Resize(output_tile_width * output_tile_height);
    ctx.d_tile_out_master_imag.Resize(output_tile_width * output_tile_height);
    ctx.d_tile_out_slave_real.Resize(output_tile_width * output_tile_height);
    ctx.d_tile_out_slave_imag.Resize(output_tile_width * output_tile_height);

    // 2d kernel sum block dimensions improve performance, seems longer x dim is always better
    // even in the unlikely scenario where azimuth window is larger - most likely due to memory access patters,
    // could use shared memory in the future as an optimization
    dim3 sum_block_dim(32, 4);
    dim3 sum_num_blocks((output_tile_width + sum_block_dim.x - 1) / sum_block_dim.x,
                        (output_tile_height + sum_block_dim.y - 1) / sum_block_dim.y);

    SimpleCoherence2DSumKernelSumSurroundings<<<sum_num_blocks, sum_block_dim, 0, ctx.stream>>>(
        ctx.data_master_norm_real.data(), ctx.d_tile_out_master_real.data(), input_tile_width, input_tile_height,
        output_tile_width, output_tile_height, coh_window.rg, coh_window.az, tile.GetXMinPad(), tile.GetXMaxPad(),
        tile.GetYMinPad(), tile.GetYMaxPad());
    CHECK_CUDA_ERRORS(cudaPeekAtLastError());

    SimpleCoherence2DSumKernelSumSurroundings<<<sum_num_blocks, sum_block_dim, 0, ctx.stream>>>(
        ctx.data_master_norm_imaginary.data(), ctx.d_tile_out_master_imag.data(), input_tile_width, input_tile_height,
        output_tile_width, output_tile_height, coh_window.rg, coh_window.az, tile.GetXMinPad(), tile.GetXMaxPad(),
        tile.GetYMinPad(), tile.GetYMaxPad());
    CHECK_CUDA_ERRORS(cudaPeekAtLastError());

    SimpleCoherence2DSumKernelSumSurroundings<<<sum_num_blocks, sum_block_dim, 0, ctx.stream>>>(
        ctx.complex_data_slave_norm_real.data(), ctx.d_tile_out_slave_real.data(), input_tile_width, input_tile_height,
        output_tile_width, output_tile_height, coh_window.rg, coh_window.az, tile.GetXMinPad(), tile.GetXMaxPad(),
        tile.GetYMinPad(), tile.GetYMaxPad());
    CHECK_CUDA_ERRORS(cudaPeekAtLastError());

    SimpleCoherence2DSumKernelSumSurroundings<<<sum_num_blocks, sum_block_dim, 0, ctx.stream>>>(
        ctx.complex_data_slave_norm_imaginary.data(), ctx.d_tile_out_slave_imag.data(), input_tile_width,
        input_tile_height, output_tile_width, output_tile_height, coh_window.rg, coh_window.az, tile.GetXMinPad(),
        tile.GetXMaxPad(), tile.GetYMinPad(), tile.GetYMaxPad());
    CHECK_CUDA_ERRORS(cudaPeekAtLastError());

    ctx.d_tile_out.Resize(output_tile_height * output_tile_width);

    thrust::transform(
        thrust_stream,
        thrust::make_zip_iterator(thrust::make_tuple(
            ctx.d_tile_out_master_real.begin(), ctx.d_tile_out_master_imag.begin(), ctx.d_tile_out_slave_real.begin(),
            ctx.d_tile_out_slave_imag.begin(), ctx.d_tile_out_slave_real_bool.begin())),
        thrust::make_zip_iterator(thrust::make_tuple(ctx.d_tile_out_master_real.end(), ctx.d_tile_out_master_imag.end(),
                                                     ctx.d_tile_out_slave_real.end(), ctx.d_tile_out_slave_imag.end(),
                                                     ctx.d_tile_out_slave_real_bool.end())),
        thrust::make_zip_iterator(thrust::make_tuple(ctx.d_tile_out.begin())), FilteredCoherenceProduct());
}

}  // namespace coherence-cuda
}  // namespace alus