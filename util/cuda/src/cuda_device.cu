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

#include "cuda_device.h"

#include <cuda_runtime_api.h>

#include "cuda_util.h"

namespace alus::cuda {

CudaDevice::CudaDevice(int device_nr, void* device_prop) : device_nr_{device_nr} {
    cudaDeviceProp* dev = reinterpret_cast<cudaDeviceProp*>(device_prop);
    cc_major_ = dev->major;
    cc_minor_ = dev->minor;
    name_ = dev->name;
    sm_count_ = dev->multiProcessorCount;
    max_threads_per_sm_ = dev->maxThreadsPerMultiProcessor;
    warp_size_ = dev->warpSize;
    total_global_memory_ = dev->totalGlobalMem;
    alignment_ = dev->textureAlignment;
}

void CudaDevice::Set() const {
    CHECK_CUDA_ERR(cudaSetDevice(device_nr_));
}

size_t CudaDevice::GetFreeGlobalMemory() const {
    Set();
    size_t total;
    size_t free;
    CHECK_CUDA_ERR(cudaMemGetInfo(&free, &total));

    return free;
}

}  // namespace alus::cuda
