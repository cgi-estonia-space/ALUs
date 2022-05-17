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

#include <cassert>
#include <cuda_runtime.h>
#include <vector>

#include "cuda_mem_arena.h"
#include "cuda_util.h"
#include "kernel_array.h"
#include "s1tbx-commons/noise_vector.h"

namespace alus::tnr {

struct alignas(4) CInt16 {
    int16_t i;
    int16_t q;
};

static_assert(sizeof(CInt16) == 4);
static_assert(alignof(CInt16) == 4);

union ComplexIntensityData {
    CInt16 input;
    float output;
};

// per thread stream + host memory buffer + device memory buffer
struct ThreadData {
    alus::PagedOrPinnedHostPtr<ComplexIntensityData> h_tile_buffer;
    cuda::MemArena dev_mem_arena;
    cudaStream_t stream = nullptr;
};
namespace device {

// Array index is intended to be used as a key.
using BurstIndexToRangeVectorMap =
    cuda::KernelArray<s1tbx::DeviceNoiseVector>;

using BurstIndexToInterpolatedRangeVectorMap = cuda::KernelArray<cuda::KernelArray<double>>;

template <typename T>
using Matrix = cuda::KernelArray<cuda::KernelArray<T>>;

/**
 * Creates a Matrix allocated on a device.
 *
 * @tparam T Any type.
 *
 * @param row_width Length of the inner KernelArrays.
 * @param row_count Amount of rows in a map.
 * @return Matrix with cells of type T.
 * @note Created map has memory allocated on device. As a result, calls to cudaFree are required. Alternatively,
 * DestroyKernelMatrix() can be called for the same result.
 */
template <typename T>
inline Matrix<T> CreateKernelMatrix(size_t row_width, size_t row_count) {
    std::vector<cuda::KernelArray<T>> h_map(row_count);
    for (auto& row : h_map) {
        row.size = row_width;
        CHECK_CUDA_ERR(cudaMalloc(&row.array, row.ByteSize()));
    }

    BurstIndexToInterpolatedRangeVectorMap d_map{nullptr, row_count};
    CHECK_CUDA_ERR(cudaMalloc(&d_map.array, d_map.ByteSize()));
    CHECK_CUDA_ERR(cudaMemcpy(d_map.array, h_map.data(), d_map.ByteSize(), cudaMemcpyHostToDevice));

    return d_map;
}

/**
 * Creates a BurstIndexToInterpolatedRangeVectorMap allocated on a device.
 *
 * @param row_width Length of the inner KernelArrays.
 * @param row_count Amount of rows in a map.
 * @return BurstIndexToInterpolatedRangeVectorMap.
 * @note Created map has memory allocated on device. As a result, calls to cudaFree are required. Alternatively,
 * DestroyBurstIndexToInterpolatedRangeVectorMap can be called for the same result.
 */
inline BurstIndexToInterpolatedRangeVectorMap CreateBurstIndexToInterpolatedRangeVectorMap(size_t row_width,
                                                                                           size_t row_count) {
    return CreateKernelMatrix<double>(row_width, row_count);
}

/**
 * Creates a BurstIndexToInterpolatedRangeVectorMap allocated on a device from vector.
 *
 * @param map Mapped vector completely allocated on the host.
 * @return  BurstIndexToInterpolatedRangeVectorMap.
 * @note Created map has memory allocated on device. As a result, calls to cudaFree are required. Alternatively,
 * DestroyBurstIndexToInterpolatedRangeVectorMap can be called for the same result.
 */
inline BurstIndexToInterpolatedRangeVectorMap CopyBurstIndexToInterpolatedRangeVectorMapToDevice(
    std::vector<std::vector<double>> map) {
    std::vector<cuda::KernelArray<double>> h_map(map.size());
    for (size_t i = 0; i < map.size(); ++i) {
        const auto& h_row = map.at(i);
        auto& d_row = h_map.at(i);
        d_row.size = h_row.size();
        CHECK_CUDA_ERR(cudaMalloc(&d_row.array, d_row.ByteSize()));
        CHECK_CUDA_ERR(cudaMemcpy(d_row.array, h_row.data(), d_row.ByteSize(), cudaMemcpyHostToDevice));
    }

    BurstIndexToInterpolatedRangeVectorMap d_map{nullptr, map.size()};
    CHECK_CUDA_ERR(cudaMalloc(&d_map.array, d_map.ByteSize()));
    CHECK_CUDA_ERR(cudaMemcpy(d_map.array, h_map.data(), d_map.ByteSize(), cudaMemcpyHostToDevice));

    return d_map;
}

/**
 * Copies matrix from device to host.
 *
 * @tparam T Any type.
 * @param d_matrix Matrix allocated on the device.
 * @return Vector of vectors.
 */
template <typename T>
inline std::vector<std::vector<T>> CopyMatrixToHost(Matrix<T> d_matrix) {
    std::vector<cuda::KernelArray<T>> temp_matrix(d_matrix.size);
    CHECK_CUDA_ERR(cudaMemcpy(temp_matrix.data(), d_matrix.array, d_matrix.ByteSize(), cudaMemcpyDeviceToHost));

    std::vector<std::vector<T>> h_matrix(d_matrix.size);
    for (size_t i = 0; i < temp_matrix.size(); ++i) {
        const auto d_row = temp_matrix.at(i);
        auto& h_row = h_matrix.at(i);
        h_row.resize(d_row.size);
        CHECK_CUDA_ERR(cudaMemcpy(h_row.data(), d_row.array, d_row.ByteSize(), cudaMemcpyDeviceToHost));
    }

    return h_matrix;
}

/**
 * Destroys a device matrix of type KernelArray oj KernelArrays.
 *
 * @tparam T Type of the matrix cell.
 * @param d_matrix Reference to matrix allocated on the device.
 */
template <typename T>
inline void DestroyKernelMatrix(cuda::KernelArray<cuda::KernelArray<T>> d_matrix) {
    std::vector<cuda::KernelArray<double>> h_matrix(d_matrix.size);
    CHECK_CUDA_ERR(cudaMemcpy(h_matrix.data(), d_matrix.array, d_matrix.ByteSize(), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERR(cudaFree(d_matrix.array));
    d_matrix.size = 0;

    for (auto array : h_matrix) {
        CHECK_CUDA_ERR(cudaFree(array.array));
    }
}

/**
 * Destroys BurstIndexToInterpolatedRangeVectorMap.
 *
 * @param d_map Map to be destroyed.
 */
inline void DestroyBurstIndexToInterpolatedRangeVectorMap(BurstIndexToInterpolatedRangeVectorMap d_map) {
    DestroyKernelMatrix(d_map);
}
}  // namespace device
}  // namespace alus::tnr