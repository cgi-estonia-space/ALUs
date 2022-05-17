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

#include <cuda_runtime_api.h>
#include <cstdint>

#include "calibration_info_computation.h"
#include "calibration_type.h"
#include "kernel_array.h"
#include "s1tbx-commons/calibration_vector_computation.h"
#include "shapes.h"

namespace alus::sentinel1calibrate {

struct CalibrationLineParameters {
    const s1tbx::CalibrationVectorComputation* calibration_vector_0;
    const s1tbx::CalibrationVectorComputation* calibration_vector_1;
    double azimuth_time;
    double mu_y;
    float* vector_0_lut;
    float* vector_1_lut;
    float* retro_vector_0_lut;
    float* retro_vector_1_lut;
};

/**
 * Kernel arguments shared by most calibration kernels
 */
struct CalibrationKernelArgs {
    sentinel1calibrate::CalibrationInfoComputation calibration_info;
    cuda::KernelArray<CalibrationLineParameters> line_parameters_array{nullptr, 0};
    Rectangle target_rectangle;
    int subset_offset_x;
    int subset_offset_y;
};

// alignas(4) gives us 1x ld.global.v2.u16 vs 2 x ld.global.u16 at PTX level
struct alignas(4) CInt16 {
    int16_t i;
    int16_t q;
};

// 2x int16_t -> 1 float calibration can reuse the same buffer for both input and output
union ComplexIntensityData {
    CInt16 iq16;
    float float32;
};

static_assert(sizeof(ComplexIntensityData) == 4, "Do not change the layout!");

struct ComplexIntensityKernelArgs {
    sentinel1calibrate::CalibrationInfoComputation calibration_info;
    cuda::KernelArray<CalibrationLineParameters> line_parameters_array{nullptr, 0};
    Rectangle target_rectangle;
    int subset_offset_y;
    cuda::KernelArray<ComplexIntensityData> d_data_array;
};

/**
 * Calculates parameters required for calibration for each line.
 */
void LaunchSetupTileLinesKernel(CalibrationKernelArgs args, cudaStream_t stream);

void LaunchComplexToFloatCalKernel(CalibrationKernelArgs args, cuda::KernelArray<ComplexIntensityData> pixels,
                                   cudaStream_t stream);

void LaunchFloatToFloatCalKernel(CalibrationKernelArgs args, cuda::KernelArray<ComplexIntensityData> pixels,
                                 cudaStream_t stream);

void PopulateLUTs(cuda::KernelArray<CalibrationLineParameters> d_line_parameters, CAL_TYPE calibration_type,
                  CAL_TYPE data_type, cudaStream_t stream);
}  // namespace alus::sentinel1calibrate