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

#include "general_constants.h"
#include "s1tbx-commons/subswath_info_computation.h"
#include "shapes.h"

namespace alus {
namespace backgeocoding {

cudaError_t LaunchDerampDemod(Rectangle rectangle, int16_t* slave_i, int16_t* slave_q, double* demod_phase,
                              double* demod_i, double* demod_q, s1tbx::DeviceSubswathInfo* sub_swath,
                              int s_burst_index);

}  // namespace backgeocoding
}  // namespace alus
