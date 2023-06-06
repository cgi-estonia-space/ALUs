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

#include "sar_segment_kernels.h"

#include <cstdint>

#include "calc_kernels.cuh"
#include "cuda_util.h"
#include "kernel_array.h"

namespace alus::sarsegment {

void ComputeDivision(cuda::KernelArray<float> vh_div_vv_dest, cuda::KernelArray<float> vh, cuda::KernelArray<float> vv,
                     size_t width, size_t height, float no_data) {
    dim3 block_size{16, 16};
    dim3 main_kernel_grid_size{static_cast<unsigned int>(width / block_size.x + 1),
                               static_cast<unsigned int>(height / block_size.y + 1)};

    math::calckernels::CalcDiv<<<main_kernel_grid_size, block_size>>>(vh, vv, width, height, vh_div_vv_dest, no_data);

    CHECK_CUDA_ERR(cudaDeviceSynchronize());
    CHECK_CUDA_ERR(cudaGetLastError());
}

void ComputeDecibel(cuda::KernelArray<float> buffer, size_t width, size_t height, float no_data) {
    dim3 block_size{16, 16};
    dim3 main_kernel_grid_size{static_cast<unsigned int>(width / block_size.x + 1),
                               static_cast<unsigned int>(height / block_size.y + 1)};
    math::calckernels::CalcDb<<<main_kernel_grid_size, block_size>>>(buffer, width, height, no_data);
    CHECK_CUDA_ERR(cudaDeviceSynchronize());
    CHECK_CUDA_ERR(cudaGetLastError());
}

}  // namespace alus::sarsegment