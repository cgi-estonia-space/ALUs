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

#include "cuda_runtime_api.h"
#include "cuda_util.h"

namespace alus {
namespace cuda {

/*
 * A simple cuda memory arena allocator
 *
 * The sized constructor or ReserveMemory can be used to preallocate a bigger device memory block.
 * Further calls just do pointer arithmetic on the original device pointer and no further
 * cudaMalloc or cudaFree calls are made.
 */

class MemArena {
public:
    explicit MemArena(size_t max_bytes) : max_byte_size_(max_bytes) {
        CHECK_CUDA_ERR(cudaMalloc(&device_arena_ptr_, max_bytes));
    }

    MemArena() = default;

    void ReserveMemory(size_t max_bytes) {
        if (device_arena_ptr_ != nullptr) {
            CHECK_CUDA_ERR(cudaFree(device_arena_ptr_));
        }
        CHECK_CUDA_ERR(cudaMalloc(&device_arena_ptr_, max_bytes));
        used_byte_size_ = 0;
        max_byte_size_ = max_bytes;
    }

    template <class T>
    T* AllocArray(size_t n_elem) {
        static_assert(std::is_pod<T>::value, "non-POD types don't make much sense with Cuda!");
        static_assert(alignof(T) <= ARENA_ALIGNMENT, "T is over-aligned?");
        return static_cast<T*>(AllocateBytes(sizeof(T) * n_elem));
    }

    void Reset() { used_byte_size_ = 0; }

    void Free() {
        used_byte_size_ = 0;
        max_byte_size_ = 0;
        CHECK_CUDA_ERR(cudaFree(device_arena_ptr_));
        device_arena_ptr_ = nullptr;
    }

    ~MemArena() noexcept { cudaFree(device_arena_ptr_); }

    void* AllocateBytes(size_t requested_bytes) {
        if (requested_bytes + used_byte_size_ > max_byte_size_) {
            std::string err_msg = "Cuda mem arena overrun: used_size = ";
            err_msg += std::to_string(used_byte_size_) + " max_size = ";
            err_msg += std::to_string(max_byte_size_) + " request = ";
            err_msg += std::to_string(requested_bytes);
            throw std::runtime_error(err_msg);
        }

        void* dev_ptr = reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(device_arena_ptr_) + used_byte_size_);

        used_byte_size_ += requested_bytes;
        const size_t alignment_overrun = used_byte_size_ % ARENA_ALIGNMENT;
        if (alignment_overrun != 0) {
            used_byte_size_ += (ARENA_ALIGNMENT - alignment_overrun);
        }
        return dev_ptr;
    }
    MemArena(MemArena&& oth) noexcept
        : device_arena_ptr_(oth.device_arena_ptr_),
          used_byte_size_(oth.used_byte_size_),
          max_byte_size_(oth.max_byte_size_) {
        oth.device_arena_ptr_ = nullptr;
        oth.used_byte_size_ = 0;
        oth.max_byte_size_ = 0;
    }

    MemArena& operator=(MemArena&& rhs) noexcept {
        if (this == &rhs) {
            return *this;
        }
        if (device_arena_ptr_) {
            cudaFree(device_arena_ptr_);
        }
        device_arena_ptr_ = rhs.device_arena_ptr_;
        used_byte_size_ = rhs.used_byte_size_;
        max_byte_size_ = rhs.max_byte_size_;

        rhs.device_arena_ptr_ = nullptr;
        rhs.max_byte_size_ = 0;
        rhs.used_byte_size_ = 0;
        return *this;
    }

    MemArena(const MemArena&) = delete;  // move only type by design
    MemArena& operator=(const MemArena&) = delete;

private:
    void* device_arena_ptr_ = nullptr;
    size_t used_byte_size_ = 0;
    size_t max_byte_size_ = 0;
    static constexpr size_t ARENA_ALIGNMENT = 128;  // TODO investigate optimal value
};
}  // namespace cuda
}  // namespace alus
