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

#include "calibration_vector_computation.h"
#include "kernel_array.h"

namespace alus {
namespace sentinel1calibrate {

struct CalibrationInfoComputation {
    double first_line_time;
    double last_line_time;
    double line_time_interval;
    int num_of_lines;
    int count;
    cuda::KernelArray<int> line_values;
    cuda::KernelArray<s1tbx::CalibrationVectorComputation> calibration_vectors;
};
}  // namespace sentinel1calibrate
}  // namespace alus