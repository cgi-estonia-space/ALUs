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

#include <cstddef>
#include <string>
#include <string_view>

namespace alus::cuda {
class CudaDevice final {
public:
    CudaDevice() = delete;
    /*
     * Parses GPU device properties from 'cudaDeviceProp' struct pointer.
     * It is opaque one here in order to not include CUDA SDK headers to host compilation.
     */
    CudaDevice(int device_nr, void* device_prop);

    void Set() const;

    [[nodiscard]] int GetDeviceNr() const { return device_nr_; }
    [[nodiscard]] std::string_view GetName() const { return name_; }
    [[nodiscard]] size_t GetCcMajor() const { return cc_major_; }
    [[nodiscard]] size_t GetCcMinor() const { return cc_minor_; }
    [[nodiscard]] size_t GetSmCount() const { return sm_count_; }
    [[nodiscard]] size_t GetMaxThreadsPerSm() const { return max_threads_per_sm_; }
    [[nodiscard]] size_t GetWarpSize() const { return warp_size_; }
    [[nodiscard]] size_t GetTotalGlobalMemory() const { return total_global_memory_; };
    [[nodiscard]] size_t GetFreeGlobalMemory() const;
    [[nodiscard]] size_t GetMemoryAlignment() const { return alignment_; }

private:
    int device_nr_;
    size_t cc_major_;
    size_t cc_minor_;
    std::string name_;
    size_t sm_count_;
    size_t max_threads_per_sm_;
    size_t warp_size_;
    size_t total_global_memory_;
    size_t alignment_;
};
}  // namespace alus::cuda
