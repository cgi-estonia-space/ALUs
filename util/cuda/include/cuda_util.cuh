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

#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/device_vector.h>
#include <vector>

#include "kernel_array.h"
#include "cuda_util.hpp"

namespace alus {
namespace cuda {

template <typename T>
KernelArray<T> GetKernelArray(thrust::device_vector<T> device_vector) {
    return {thrust::raw_pointer_cast(device_vector.data()), device_vector.size()};
}

template <typename T>
KernelArray<T> GetKernelArray(std::vector<T> &data_vector) {
    return {data_vector.data(), data_vector.size()};
}

/**
 * Copies KernelArray from host to device.
 *
 * @tparam T Type of KernelArray.
 * @param host_kernel_array Kernel array allocated to host memory.
 * @param d_array NULL pointer which will point to memory allocated on device. Allocation is being done in this function.
 * @return Returns new KernelArray of type T with KernelArray.array member pointing to allocated and filled <i>d_array</i> parameter.
 * @todo Currently works only with simple data types which do not require manual copying between host and device, i.e. T should not be any array or collection.
 */
template <typename T>
KernelArray<T> CopyKernelArrayToDevice(T *&d_array, KernelArray<T> host_kernel_array) {
    CHECK_CUDA_ERR(cudaMalloc((void **)&d_array, sizeof(T) * host_kernel_array.size));
    CHECK_CUDA_ERR(cudaMemcpy(d_array, host_kernel_array.array, sizeof(T) * host_kernel_array.size, cudaMemcpyHostToDevice));
    return {d_array, host_kernel_array.size};
}

/**
 * Copies KernelArray from device to host.
 *
 * @attention This function <b>does not</b> free device memory. This is intended behaviour as the device kernel array
 * might be used in subsequent CUDA operations.
 *
 * @tparam T Type of KernelArray.
 * @param device_kernel_array Kernel array allocated to device memory.
 * @param h_array Pointer to already allocated empty array on host memory.
 * @return Returns new KernelArray of type T with KernelArray.array member pointing to filled <i>h_array</i> parameter.
 * @todo Currently works only with simple data types which do not require manual copying between host and device, i.e. T should not be any array or collection.
 */
template <typename T>
KernelArray<T> CopyKernelArrayToHost(KernelArray<T> device_kernel_array, T *h_array) {
    CHECK_CUDA_ERR(
        cudaMemcpy(h_array, device_kernel_array.array, sizeof(T) * device_kernel_array.size, cudaMemcpyDeviceToHost));
    return {h_array, device_kernel_array.size};
}
}  // namespace cuda
}  // namespace alus
