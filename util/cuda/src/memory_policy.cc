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

#include "memory_policy.h"

namespace alus::cuda {

MemoryAllocationForecast::MemoryAllocationForecast(size_t alignment) : alignment_{alignment} {}

void MemoryAllocationForecast::Add(size_t bytes) {
    if (bytes % alignment_ != 0) {
        forecast_ += ((bytes / alignment_) + 1) * alignment_;
    } else {
        forecast_ += bytes;
    }
}

MemoryFitPolice::MemoryFitPolice(const CudaDevice& device, size_t percentage_allowed)
    : device_{device}, percentage_{percentage_allowed}, total_memory_{device_.GetTotalGlobalMemory()} {}

bool MemoryFitPolice::CanFit(size_t bytes) const {
    const auto allowed_memory =
        static_cast<size_t>(total_memory_ * (static_cast<double>(percentage_) / 100.0));
    if (bytes > allowed_memory) {
        return false;
    }

    if (device_.GetFreeGlobalMemory() < bytes) {
        return false;
    }

    return true;
}

}  // namespace alus::cuda