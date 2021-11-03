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
#pragma once

#include <cuda_runtime.h>

#include <cublas_v2.h>

#include <thrust/complex.h>
#include <thrust/device_vector.h>
#include <thrust/tuple.h>

#include "band_params.h"
#include "coh_tile.h"
#include "coh_window.h"
#include "cuda_ptr.h"

namespace alus {
namespace coherence_cuda {

// Per thread gpu buffers, handles to avoid reallocating for each tile calculation
struct ThreadContext {
    // main coherence buffers
    cuda::DeviceBuffer<float> d_band_master_real;
    cuda::DeviceBuffer<float> d_band_master_imag;
    cuda::DeviceBuffer<float> d_band_slave_real;
    cuda::DeviceBuffer<float> d_band_slave_imag;
    cuda::DeviceBuffer<bool> d_tile_out_slave_real_bool;

    cuda::DeviceBuffer<float> complex_data_slave_norm_real;
    cuda::DeviceBuffer<float> complex_data_slave_norm_imaginary;
    cuda::DeviceBuffer<float> data_master_norm_real;
    cuda::DeviceBuffer<float> data_master_norm_imaginary;

    cuda::DeviceBuffer<float> d_tile_out_master_real;
    cuda::DeviceBuffer<float> d_tile_out_master_imag;
    cuda::DeviceBuffer<float> d_tile_out_slave_real;
    cuda::DeviceBuffer<float> d_tile_out_slave_imag;

    cuda::DeviceBuffer<float> d_tile_out;

    // flat earth phase subtraction buffers
    cuda::DeviceBuffer<double> d_range_axis;
    cuda::DeviceBuffer<double> d_azimuth_axis;
    cuda::DeviceBuffer<double> d_x_xpows_ones_t;
    cuda::DeviceBuffer<double> d_y_ypows_ones_t;

    cuda::DeviceBuffer<double> d_y_ones_ypows_t;
    cuda::DeviceBuffer<double> d_x_ones_xpows_t;
    cuda::DeviceBuffer<double> d_y_ones;
    cuda::DeviceBuffer<double> d_x_ones;
    cuda::DeviceBuffer<double> d_y_ones_coefs_t;
    cuda::DeviceBuffer<double> d_flat_earth_phase;

    // CPU <-> GPU transfer buffer
    PagedOrPinnedHostPtr<float> h_buffer;

    // cuda handles
    cudaStream_t stream = nullptr;
    cublasHandle_t handle = nullptr;
};

class CoherenceComputation {
public:
    //  flat earth phase substraction
    thrust::device_vector<double> d_x_pows_;
    thrust::device_vector<double> d_y_pows_;
    thrust::device_vector<double> d_coefs_;
    thrust::device_vector<double> d_ones_;
    size_t d_ones_size_ = 0;
    bool subtract_flat_earth_ = false;

    /**
     * Generates A (Ax = b). Specific function for internal use.
     * Normalizes input vectors x and y and generates element-wise powers.
     * @param cuBLAS library handle
     * @param d_x x side device vector of size n
     * @param d_y y side device vector of size n
     * @param d_out device vector into which results will be saved
     * @param x_norm_min normalization minimum for x
     * @param x_norm_max normalization maximum for x
     * @param y_norm_min normalization minimum for y
     * @param y_norm_max normalization maximum for y
     */
    void GenerateA(cublasHandle_t handle, thrust::device_vector<double>& d_x, thrust::device_vector<double>& d_y,
                   thrust::device_vector<double>& d_out, double x_norm_min, double x_norm_max, double y_norm_min,
                   double y_norm_max);

    /**
     * Transpose matrix A, wrapper for cuBLAS cublasDgeam
     * @param cuBLAS library handle
     * @param A input matrix
     * @param C result matrix
     * @param m number of rows of matrix op(A) and C (op(A) == CUBLAS_OP_T)
     * @param n number of columns of C.
     */
    static void TransposeA(cublasHandle_t handle, const double* A, double* C, const int m, const int n);

    /**
     * Multiplies matrix A by transposed matrix A, wrapper for cuBLAS cublasDgemm
     * @param cuBLAS library handle
     * @param A input matrix
     * @param C result matrix
     * @param m number of rows of matrix op(A) and C (op(A) == CUBLAS_OP_N)
     * @param n number of columns of matrix op(A) and C (op(A) == CUBLAS_OP_T)
     */
    static void MatMulATransposeA(cublasHandle_t handle, const double* A, double* C, const int m, const int n);

    /**
     * Multiplies matrix A by transposed matrix B, wrapper for cuBLAS cublasDgemm
     * @param cuBLAS library handle
     * @param A input matrix
     * @param B input matrix
     * @param C result matrix
     * @param m number of rows of matrix op(A) and C (op(A) == CUBLAS_OP_N)
     * @param k number of columns of op(A) and rows of op(B) (op(B) == CUBLAS_OP_T)
     * @param n number of columns of matrix op(B) and C
     */
    static void MatMulATransposeB(cublasHandle_t handle, const double* A, const double* B, double* C, const int m,
                                  const int k, const int n);

    /**
     * Multiplies matrix A by matrix B, wrapper for cuBLAS cublasDgemm
     * @param cuBLAS library handle
     * @param A input matrix
     * @param B input matrix
     * @param C result matrix
     * @param m number of rows of matrix op(A) and C (op(A) == CUBLAS_OP_N)
     * @param k number of columns of op(A) and rows of op(B) (op(B) == CUBLAS_OP_N)
     * @param n number of columns of matrix op(B) and C
     */
    static void MatMulAB(cublasHandle_t handle, const double* A, const double* B, double* C, const int m, const int k,
                         const int n);

    /**
     * fills d_vector with generated values (check numpy linspace)
     * @param min start value of the sequence
     * @param max end value of the sequence
     * @param d_vector device vector to hold results
     */
    static void Linspance(double min, double max, cuda::CudaPtr<double>& d_vector, cudaStream_t stream);

    /**
     * Calculations of dependencies which are later needed by tiles
     * for coherence chain these are mostly related to calculating coefficients
     */
    void LaunchCoherencePreTileCalc(std::vector<int>& x_pows, std::vector<int>& y_pows,
                                    std::tuple<std::vector<int>, std::vector<int>>& position_lines_pixels,
                                    std::vector<double>& generate_y, const BandParams& band_params);

    /**
     * Calculate coherence for tile
     * @param tile input tile
     * @param data flat input data bands one after another, row major order
     * @param data_out results container, filled during calculations
     * @param coh_window coherence azimuth and range
     * @param band_params
     */
    void LaunchCoherence(const CohTile& tile, ThreadContext& buffers, const CohWindow& coh_window,
                         const BandParams& band_params);
};
}  // namespace coherence_cuda
}  // namespace alus