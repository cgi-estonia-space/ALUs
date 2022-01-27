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

#include "cuda_runtime_api.h"
#include "cuda_util.h"

#include <iostream>

namespace alus {  // NOLINT TODO: concatenate namespace and remove nolint after migrating to cuda 11+
namespace cuda {

/**
    Small wrappers around cudaMemcpy using logical size over bytesize, verifying that pointer types match
    and the T type is a POD
 */
template <class T>
void CopyAsyncH2D(T* d_dest, T* h_src, cudaStream_t stream) {
    static_assert(std::is_pod<T>::value, "non-POD types don't make much sense with Cuda!");
    CHECK_CUDA_ERR(cudaMemcpyAsync(d_dest, h_src, sizeof(T), cudaMemcpyHostToDevice, stream));
}

template <class T>
void CopyAsyncD2H(T* h_dest, T* d_src, cudaStream_t stream) {
    static_assert(std::is_pod<T>::value, "non-POD types don't make much sense with Cuda!");
    CHECK_CUDA_ERR(cudaMemcpyAsync(h_dest, d_src, sizeof(T), cudaMemcpyDeviceToHost, stream));
}

template <class T>
void CopyArrayAsyncH2D(T* d_dest, T* h_src, size_t n_elem, cudaStream_t stream) {
    static_assert(std::is_pod<T>::value, "non-POD types don't make much sense with Cuda!");
    CHECK_CUDA_ERR(cudaMemcpyAsync(d_dest, h_src, sizeof(T) * n_elem, cudaMemcpyHostToDevice, stream));
}

template <class T>
void CopyArrayAsyncD2H(T* h_dest, T* d_src, size_t n_elem, cudaStream_t stream) {
    static_assert(std::is_pod<T>::value, "non-POD types don't make much sense with Cuda!");
    CHECK_CUDA_ERR(cudaMemcpyAsync(h_dest, d_src, sizeof(T) * n_elem, cudaMemcpyDeviceToHost, stream));
}

template <class T>
void CopyH2D(T* d_dest, T* h_src) {
    static_assert(std::is_pod<T>::value, "non-POD types don't make much sense with Cuda!");
    CHECK_CUDA_ERR(cudaMemcpy(d_dest, h_src, sizeof(T), cudaMemcpyHostToDevice));
}

template <class T>
void CopyD2H(T* h_dest, T* d_src) {
    static_assert(std::is_pod<T>::value, "non-POD types don't make much sense with Cuda!");
    CHECK_CUDA_ERR(cudaMemcpy(h_dest, d_src, sizeof(T), cudaMemcpyDeviceToHost));
}

template <class T>
void CopyArrayH2D(T* d_dest, T* h_src, size_t n_elem) {
    static_assert(std::is_pod<T>::value, "non-POD types don't make much sense with Cuda!");
    CHECK_CUDA_ERR(cudaMemcpy(d_dest, h_src, sizeof(T) * n_elem, cudaMemcpyHostToDevice));
}

template <class T>
void CopyArrayD2H(T* h_dest, T* d_src, size_t n_elem) {
    static_assert(std::is_pod<T>::value, "non-POD types don't make much sense with Cuda!");
    CHECK_CUDA_ERR(cudaMemcpy(h_dest, d_src, sizeof(T) * n_elem, cudaMemcpyDeviceToHost));
}
}  // namespace cuda
}  // namespace alus