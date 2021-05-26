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

#include <algorithm>
#include <cstddef>

#include <cuda_runtime.h>

#include "cuda_util.h"

namespace alus {
namespace cuda {

inline int DivUp(int a, int b) { return (a + b - 1) / b; }

struct LaunchConfig2D {
    dim3 grid_size;   // Block count in grid, 1st argument between kernel conf '<<<>>>'
    dim3 block_size;  // Threads per block, 2nd argument between kernel conf '<<<>>>'
    dim3 virtual_thread_count;
};

/**
 * Calculates Kernel launch parameters based on CUDA API cudaOccupancyMaxPotentialBlockSize()
 *
 * Implementation copied from TensorFlow project https://github.com/tensorflow/tensorflow version 2.4.
 * Original implementation can be found in core/util/gpu_launch_config.h as GetGpu3DLaunchConfig().
 *
 * @tparam KernelFunc
 * @param x_size Width/Dimension of an array of data to be processed
 * @param y_size Height/Dimension of an array of data to be processed
 * @param kernel Device function symbol
 * @param dynamic_shared_memory_size Per-block dynamic shared memory usage intended, in bytes.
 *                                   0(default) means no shared memory used.
 * @param block_size_limit The maximum block size func is designed to work with. 0 means no limit.
 * @return Most optimal grid and block sizes
 */
template <typename DeviceFunc>
LaunchConfig2D GetLaunchConfig2D(int x_size, int y_size, DeviceFunc dev_func, size_t dynamic_shared_memory_size = 0,
                                 int block_size_limit = 0) {
    if (x_size <= 0 || y_size <= 0) {
        return {};
    }

    int dev{};
    CHECK_CUDA_ERR(cudaGetDevice(&dev));

    int x_thread_limit{};
    CHECK_CUDA_ERR(cudaDeviceGetAttribute(&x_thread_limit, cudaDevAttrMaxBlockDimX, dev));
    int y_thread_limit{};
    CHECK_CUDA_ERR(cudaDeviceGetAttribute(&y_thread_limit, cudaDevAttrMaxBlockDimY, dev));
    int x_grid_limit{};
    CHECK_CUDA_ERR(cudaDeviceGetAttribute(&x_grid_limit, cudaDevAttrMaxGridDimX, dev));
    int y_grid_limit{};
    CHECK_CUDA_ERR(cudaDeviceGetAttribute(&y_grid_limit, cudaDevAttrMaxGridDimY, dev));

    int block_count{};
    int thread_per_block{};
    cudaError_t err = cudaOccupancyMaxPotentialBlockSize(&block_count, &thread_per_block, dev_func,
                                                         dynamic_shared_memory_size, block_size_limit);
    CHECK_CUDA_ERR(err);

    int threads_x = std::min({x_size, thread_per_block, x_thread_limit});
    int threads_y = std::min({y_size, std::max(thread_per_block / threads_x, 1), y_thread_limit});

    int blocks_x = std::min({block_count, DivUp(x_size, threads_x), x_grid_limit});
    int blocks_y = std::min({DivUp(block_count, blocks_x), DivUp(y_size, threads_y), y_grid_limit});

    LaunchConfig2D config{};
    config.grid_size = dim3{static_cast<unsigned int>(blocks_x), static_cast<unsigned int>(blocks_y), 1};
    config.block_size = dim3{static_cast<unsigned int>(threads_x), static_cast<unsigned int>(threads_y), 1};
    config.virtual_thread_count = dim3{static_cast<unsigned int>(x_size), static_cast<unsigned int>(y_size)};
    return config;
}

template <typename DeviceFunc>
double GetOccupancyPercentageFor(DeviceFunc dev_func, dim3 block_size) {
    int device{};
    CHECK_CUDA_ERR(cudaGetDevice(&device));
    int warp_size{};
    CHECK_CUDA_ERR(cudaDeviceGetAttribute(&warp_size, cudaDevAttrWarpSize, device));
    int max_threads_per_multi_processor{};
    CHECK_CUDA_ERR(
        cudaDeviceGetAttribute(&max_threads_per_multi_processor, cudaDevAttrMaxThreadsPerMultiProcessor, device));

    int total_threads_blocks = block_size.x * block_size.y;
    int num_blocks;
    CHECK_CUDA_ERR(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks, dev_func, total_threads_blocks, 0));

    int active_warps = num_blocks * total_threads_blocks / warp_size;
    int max_warps = max_threads_per_multi_processor / warp_size;

    return (static_cast<double>(active_warps) / max_warps) * 100;
}

/**
 * Helper class for range-based for loop using 'delta' increments.
 * Should be used via helper functions GpuGridRange*()
 *
 * Ported from tensorflow/core/util/gpu_device_functions.h
 */
template <typename T>
class GpuGridRange {
    struct Iterator {
        __device__ Iterator(T index, T delta) : index_(index), delta_(delta) {}
        __device__ T operator*() const { return index_; }
        __device__ Iterator& operator++() {
            index_ += delta_;
            return *this;
        }
        __device__ bool operator!=(const Iterator& other) const {
            bool greater = index_ > other.index_;
            bool less = index_ < other.index_;
            // Anything past an end iterator (delta_ == 0) is equal.
            // In range-based for loops, this optimizes to 'return less'.
            if (!other.delta_) {
                return less;
            }
            if (!delta_) {
                return greater;
            }
            return less || greater;
        }

    private:
        T index_;
        const T delta_;
    };

public:
    __device__ GpuGridRange(T begin, T delta, T end)
        : begin_(begin), delta_(delta), end_(end) {}

    __device__ Iterator begin() const { return Iterator{begin_, delta_}; }
    __device__ Iterator end() const { return Iterator{end_, 0}; }

private:
    T begin_;
    T delta_;
    T end_;
};

// Helper to visit indices in the range 0 <= i < count, using the x-coordinate
// of the global thread index. That is, each index i is visited by all threads
// with the same x-coordinate.
// Usage: for(int i : GpuGridRangeX(count)) { visit(i); }
template <typename T>
__device__ GpuGridRange<T> GpuGridRangeX(T count) {
    return GpuGridRange<T>(blockIdx.x * blockDim.x + threadIdx.x,
                                   gridDim.x * blockDim.x, count);
}

// Helper to visit indices in the range 0 <= i < count using the y-coordinate.
// Usage: for(int i : GpuGridRangeY(count)) { visit(i); }
template <typename T>
__device__ GpuGridRange<T> GpuGridRangeY(T count) {
    return GpuGridRange<T>(blockIdx.y * blockDim.y + threadIdx.y,
                                   gridDim.y * blockDim.y, count);
}

// Helper to visit indices in the range 0 <= i < count using the z-coordinate.
// Usage: for(int i : GpuGridRangeZ(count)) { visit(i); }
template <typename T>
__device__ GpuGridRange<T> GpuGridRangeZ(T count) {
    return GpuGridRange<T>(blockIdx.z * blockDim.z + threadIdx.z,
                                   gridDim.z * blockDim.z, count);
}

#define PRINT_DEVICE_FUNC_OCCUPANCY(dev_func, block_size)                                                         \
    std::cout << #dev_func << " occupancy " << alus::cuda::GetOccupancyPercentageFor(dev_func, block_size) << "%" \
              << std::endl

}  // namespace cuda
}  // namespace alus