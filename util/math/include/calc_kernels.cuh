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

#include <cmath>
#include <cstddef>

#include "kernel_array.h"

namespace alus::math::calckernels {

__global__ void CalcDb(cuda::KernelArray<float> buffer, size_t w, size_t h, float no_data_value = nanf(""));

/**
 *
 * @param dividend
 * @param divisor
 * @param w Raster width, all buffers shall have the same dimensions.
 * @param h Raster height, all buffers shall have the same dimensions.
 * @param result When divisor is 0, the result will be 0. When dividend or divisor isnan() == true -> 'no_data_value'
 * @param no_data_value Dividend or divisor pixels that contain this will be resulting in 'no_data_value' for result.
 * Used only when not equal to NaN.
 */
__global__ void CalcDiv(cuda::KernelArray<float> dividend, cuda::KernelArray<float> divisor, size_t w, size_t h,
                        cuda::KernelArray<float> result, float no_data_value = nanf(""));

}  // namespace alus::math::calckernels