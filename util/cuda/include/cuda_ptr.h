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

namespace alus {
namespace cuda {

/**
 * Use this class if you want to make sure that a cuda pointer gets freed, if it leaves scope. There is also free() and
 * resize() for those moments, when you need to micromanage the memory to make the most of it.
 * @tparam T The type of device pointer. For readibility.
 */
template <typename T>
class CudaPtr {
public:
    explicit CudaPtr(size_t elem_count) { CHECK_CUDA_ERR(cudaMalloc((void**)&device_ptr_, elem_count*sizeof(T))); }
    ~CudaPtr() { free(); }
    CudaPtr(const CudaPtr&) = delete;  // class does not support copying(and moving)
    CudaPtr& operator=(const CudaPtr&) = delete;

    void free() {
        if (device_ptr_ != nullptr) {
            cudaFree(device_ptr_);
            device_ptr_ = nullptr;
        }
    }

    void Reallocate(size_t size) {
        free();
        CHECK_CUDA_ERR(cudaMalloc((void**)&device_ptr_, size));
    }
    T* Get(){
        return device_ptr_;
    }

private:
    T* device_ptr_ = nullptr;
};

}  // namespace cuda
}  // namespace alus
