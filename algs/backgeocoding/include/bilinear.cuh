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
#include <device_launch_parameters.h>

namespace alus {
namespace backgeocoding{

cudaError_t LaunchBilinearInterpolation(
                        dim3 grid_size,
                        dim3 block_size,
                        double *x_pixels,
                        double *y_pixels,
                        double *demod_phase,
                        double *demod_i,
                        double *demod_q,
                        int *int_params,
                        double double_params,
                        float *results_i,
                        float *results_q);

} //namespace
} //namespace
