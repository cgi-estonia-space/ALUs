#pragma once

#include "cuda_runtime_api.h"
#include "cuda_util.h"

#include <iostream>

namespace alus {
namespace cuda {

/**
    Small wrappers around cudaMemcpy using logical size over bytesize, verifying that pointer types match
    and the T type is a POD
 */
template <class T>
void copyAsyncH2D(T* d_dest, T* h_src, cudaStream_t stream) {
    static_assert(std::is_pod<T>::value, "non-POD types don't make much sense with Cuda!");
    CHECK_CUDA_ERR(cudaMemcpyAsync(d_dest, h_src, sizeof(T), cudaMemcpyHostToDevice, stream));
}

template <class T>
void copyAsyncD2H(T* h_dest, T* d_src, cudaStream_t stream) {
    static_assert(std::is_pod<T>::value, "non-POD types don't make much sense with Cuda!");
    CHECK_CUDA_ERR(cudaMemcpyAsync(h_dest, d_src, sizeof(T), cudaMemcpyDeviceToHost, stream));
}

template <class T>
void copyArrayAsyncH2D(T* d_dest, T* h_src, size_t n_elem, cudaStream_t stream) {
    static_assert(std::is_pod<T>::value, "non-POD types don't make much sense with Cuda!");
    CHECK_CUDA_ERR(cudaMemcpyAsync(d_dest, h_src, sizeof(T) * n_elem, cudaMemcpyHostToDevice, stream));
}

template <class T>
void copyArrayAsyncD2H(T* h_dest, T* d_src, size_t n_elem, cudaStream_t stream) {
    static_assert(std::is_pod<T>::value, "non-POD types don't make much sense with Cuda!");
    CHECK_CUDA_ERR(cudaMemcpyAsync(h_dest, d_src, sizeof(T) * n_elem, cudaMemcpyDeviceToHost, stream));
}

template <class T>
void copyH2D(T* d_dest, T* h_src) {
    static_assert(std::is_pod<T>::value, "non-POD types don't make much sense with Cuda!");
    CHECK_CUDA_ERR(cudaMemcpy(d_dest, h_src, sizeof(T), cudaMemcpyHostToDevice));
}

template <class T>
void copyD2H(T* h_dest, T* d_src) {
    static_assert(std::is_pod<T>::value, "non-POD types don't make much sense with Cuda!");
    CHECK_CUDA_ERR(cudaMemcpy(h_dest, d_src, sizeof(T), cudaMemcpyDeviceToHost));
}

template <class T>
void copyArrayH2D(T* d_dest, T* h_src, size_t n_elem) {
    static_assert(std::is_pod<T>::value, "non-POD types don't make much sense with Cuda!");
    CHECK_CUDA_ERR(cudaMemcpy(d_dest, h_src, sizeof(T) * n_elem, cudaMemcpyHostToDevice));
}

template <class T>
void copyArrayD2H(T* h_dest, T* d_src, size_t n_elem) {
    static_assert(std::is_pod<T>::value, "non-POD types don't make much sense with Cuda!");
    CHECK_CUDA_ERR(cudaMemcpy(h_dest, d_src, sizeof(T) * n_elem, cudaMemcpyDeviceToHost));
}
}  // namespace cuda
}  // namespace alus