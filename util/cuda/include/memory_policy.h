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

#include "cuda_device.h"

namespace alus::cuda {

class MemoryAllocationForecast final {
public:
    MemoryAllocationForecast() = delete;
    explicit MemoryAllocationForecast(size_t alignment);

    void Add(size_t bytes);
    [[nodiscard]] size_t Get() const { return forecast_; }

private:
    size_t alignment_;
    size_t forecast_{};
};

class MemoryFitPolice final {
public:
    MemoryFitPolice() = delete;
    MemoryFitPolice(const CudaDevice& device, size_t percentage_allowed);

    [[nodiscard]] bool CanFit(size_t bytes) const;

private:
    const CudaDevice& device_;
    const size_t percentage_;
    const size_t total_memory_;
};

}  // namespace alus::cuda