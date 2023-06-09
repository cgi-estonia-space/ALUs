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

#include <cstdint>

#include "kernel_array.h"

namespace alus::sarsegment {

void ComputeDivision(cuda::KernelArray<float> vh_div_vv_dest, cuda::KernelArray<float> vh, cuda::KernelArray<float> vv,
                     size_t width, size_t height, float no_data);

void ComputeDecibel(cuda::KernelArray<float> buffer, size_t width, size_t height, float no_data);

void Despeckle(cuda::KernelArray<float> in, cuda::KernelArray<float> despeckle_buffer, size_t width, size_t height,
               size_t window);

}  // namespace alus::sarsegment