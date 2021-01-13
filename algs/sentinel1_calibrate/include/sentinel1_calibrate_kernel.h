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

#include "calibration_info_computation.h"
#include "calibration_type.h"
#include "calibration_vector_computation.h"
#include "kernel_array.h"
#include "shapes.h"

namespace alus {
namespace sentinel1calibrate {

struct CalibrationLineParameters {
    const s1tbx::CalibrationVectorComputation* calibration_vector_0;
    const s1tbx::CalibrationVectorComputation* calibration_vector_1;
    double azimuth_time;
    double mu_y;
    float* vector_0_lut;
    float* vector_1_lut;
    float* retro_vector_0_lut;
    float* retor_vector_1_lut;
};

struct CalibrationPixelParameters {
    double dn;
    int64_t pixel_index;
    double mu_x;
    double lut_val;
    double retro_lut_val{1.0};
    double calibration_factor;
    int source_index;
    double phase_term;
};

/**
 * Kernel arguments shared by most calibration kernels
 */
struct CalibrationKernelArgs {
    sentinel1calibrate::CalibrationInfoComputation calibration_info;
    cuda::KernelArray<CalibrationLineParameters> line_parameters_array{nullptr, 0};
    Rectangle target_rectangle;
    CAL_TYPE data_type;
    int subset_offset_y;
    int subset_offset_x;
    cuda::KernelArray<float> source_data_1;
    cuda::KernelArray<float> source_data_2;
};

/**
 * Calculates parameters required for calibration for each line.
 */
void LaunchSetupTileLinesKernel(CalibrationKernelArgs args);

void LaunchCalculatePixelParamsKernel(CalibrationKernelArgs args,
                                             cuda::KernelArray<CalibrationPixelParameters> pixel_parameters);

void LaunchAmplitudeKernel(CalibrationKernelArgs args,
                                  cuda::KernelArray<CalibrationPixelParameters> pixel_parameters,
                                  cuda::KernelArray<double> calibration_values);

void LaunchIntensityWithoutRetroKernel(CalibrationKernelArgs args,
                                              cuda::KernelArray<CalibrationPixelParameters> pixel_parameters,
                                              cuda::KernelArray<double> calibration_values);

void LaunchIntensityWithRetroKernel(CalibrationKernelArgs args,
                                           cuda::KernelArray<CalibrationPixelParameters> pixel_parameters,
                                           cuda::KernelArray<double> calibration_values);

void LaunchImaginaryKernel(CalibrationKernelArgs args,
                                  cuda::KernelArray<CalibrationPixelParameters> pixel_parameters,
                                  cuda::KernelArray<double> calibration_values);

void LaunchComplexIntensityKernel(CalibrationKernelArgs args,
                                  cuda::KernelArray<CalibrationPixelParameters> pixel_parameters,
                                  cuda::KernelArray<double> calibration_values);

void LaunchRealKernel(CalibrationKernelArgs args,
                             cuda::KernelArray<CalibrationPixelParameters> pixel_parameters,
                             cuda::KernelArray<double> calibration_values);

void LaunchIntensityDBKernel(CalibrationKernelArgs args,
                                    cuda::KernelArray<CalibrationPixelParameters> pixel_parameters,
                                    cuda::KernelArray<double> calibration_values);

void LaunchAdjustForCompexOutputKernel(CalibrationKernelArgs args,
                                              cuda::KernelArray<CalibrationPixelParameters> pixel_parameters,
                                              cuda::KernelArray<double> calibration_values);

void PopulateLUTs(cuda::KernelArray<CalibrationLineParameters> d_line_parameters, CAL_TYPE calibration_type,
                  CAL_TYPE data_type);
}  // namespace sentinel1calibrate
}  // namespace alus