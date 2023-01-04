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
#include <cufft.h>
#include <memory>

namespace alus::palsar {

// Wrappers around cuda APIs for exception safety
struct CudaMallocDeleter {
    void operator()(void* d_ptr) { cudaFree(d_ptr); }
};

template <class T>
using CudaMallocTypeCleanup = std::unique_ptr<T, CudaMallocDeleter>;

using CudaMallocCleanup = std::unique_ptr<void, CudaMallocDeleter>;

class CufftPlanCleanup {
public:
    explicit CufftPlanCleanup(cufftHandle plan) : plan_(plan) {}
    CufftPlanCleanup(const CufftPlanCleanup&) = delete;
    CufftPlanCleanup& operator=(const CufftPlanCleanup&) = delete;
    ~CufftPlanCleanup() { cufftDestroy(plan_); }

private:
    cufftHandle plan_;
};
}  // namespace alus::palsar
