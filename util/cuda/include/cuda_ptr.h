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

#include "cuda_util.h"

namespace alus::cuda {

/**
 * Use this class if you want to make sure that a cuda pointer gets freed, if it leaves scope. There is also free() and
 * resize() for those moments, when you need to micromanage the memory to make the most of it.
 * @tparam T The type of device pointer. For readibility.
 */
template <typename T>
class CudaPtr {
public:
    explicit CudaPtr(size_t elem_count) : elem_count_(elem_count), capacity_(elem_count) {
        CHECK_CUDA_ERR(cudaMalloc((void**)&device_ptr_, elem_count * sizeof(T)));
    }
    CudaPtr() = default;
    ~CudaPtr() { free(); }
    CudaPtr(const CudaPtr&) = delete;  // class does not support copying
    CudaPtr& operator=(const CudaPtr&) = delete;
    CudaPtr(CudaPtr<T>&& oth) noexcept
        : device_ptr_(oth.device_ptr_), elem_count_(oth.elem_count_), capacity_(oth.capacity_) {
        oth.device_ptr_ = nullptr;
        oth.elem_count_ = oth.capacity_ = 0;
    }

    CudaPtr& operator=(CudaPtr<T>&& rhs) noexcept {
        if (&rhs != this) {
            if (device_ptr_) {
                free();
            }
            device_ptr_ = rhs.device_ptr_;
            elem_count_ = rhs.elem_count_;
            capacity_ = rhs.capacity_;
            rhs.device_ptr_ = nullptr;
            rhs.elem_count_ = rhs.capacity_ = 0;
        }
        return *this;
    }

    void free() {  // NOLINT
        if (device_ptr_ != nullptr) {
            REPORT_WHEN_CUDA_ERR(cudaFree(device_ptr_));
            device_ptr_ = nullptr;
            elem_count_ = 0;
            capacity_ = 0;
        }
    }

    [[nodiscard]] size_t GetElemCount() const { return elem_count_; }

    void Resize(size_t size) {
        // avoid reallocating if possible, works better with streams
        if (size > capacity_) {
            CHECK_CUDA_ERR(cudaFree(device_ptr_));
            CHECK_CUDA_ERR(cudaMalloc(&device_ptr_, size * sizeof(T)));
            elem_count_ = size;
            capacity_ = size;
        } else {
            elem_count_ = size;
        }
    }
    T* Get() { return device_ptr_; }

    // basic STL interface, useful for moving from device_vector to this
    T* begin() { return device_ptr_; }                         // NOLINT
    T* end() { return device_ptr_ + elem_count_; }             // NOLINT
    [[nodiscard]] size_t size() const { return elem_count_; }  // NOLINT
    T* data() { return device_ptr_; }                          // NOLINT

private:
    T* device_ptr_ = nullptr;
    size_t elem_count_ = 0;
    size_t capacity_ = 0;
};

// a better name is buffer as it contains size and capacity information as well, TODO refactor it later
template <class T>
using DeviceBuffer = CudaPtr<T>;

}  // namespace alus::cuda
