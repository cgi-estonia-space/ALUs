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

#include <cstddef>
#include <cstdint>

#include "calibration_info_computation.h"
#include "calibration_type.h"
#include "general_constants.h"
#include "kernel_array.h"
#include "sentinel1_calibrate_kernel.h"

#include "math_utils.cuh"

namespace alus {
namespace sentinel1calibrate {

inline __host__ __device__ size_t GetCalibrationVectorIndexImpl(int y, int count, const int* line_values) {
    const auto index = mathutils::FindFirstGreaterElement(y, count, line_values);

    return mathutils::ChooseOne(index == snapengine::constants::INVALID_INDEX,
                                static_cast<int64_t>(snapengine::constants::INVALID_INDEX), index - 1);
}

inline __device__ __host__ void SetupTileLineImpl(int y, CalibrationKernelArgs& args,
                                                  CalibrationLineParameters& line_parameters) {
    const auto calibration_vector_index = GetCalibrationVectorIndexImpl(y, args.calibration_info.line_values.size,
                                                                        args.calibration_info.line_values.array);
    line_parameters.calibration_vector_0 = &args.calibration_info.calibration_vectors.array[calibration_vector_index];
    line_parameters.calibration_vector_1 =
        &args.calibration_info.calibration_vectors.array[calibration_vector_index + 1];

    line_parameters.azimuth_time =
        args.calibration_info.first_line_time + (args.subset_offset_y + y) * args.calibration_info.line_time_interval;
    line_parameters.mu_y =
        (line_parameters.azimuth_time - line_parameters.calibration_vector_0->time_mjd) /
        (line_parameters.calibration_vector_1->time_mjd - line_parameters.calibration_vector_0->time_mjd);
}

inline __device__ __host__ int64_t GetPixelIndexImpl(int x,
                                                     const s1tbx::CalibrationVectorComputation* calibration_vector) {
    auto index =
        mathutils::BinarySearch(x, static_cast<int>(calibration_vector->array_size), calibration_vector->pixels);
    index = mathutils::ChooseOne(index < 0, index * (-1) - 2, index);
    index = mathutils::ChooseOne(index >= static_cast<int>(calibration_vector->array_size - 1), index - 1, index);

    return index;
}

inline __device__ __host__ double CalculateLutValImpl(const CalibrationLineParameters& line_parameters, double mu_x,
                                                      int64_t pixel_index) {
    const auto* vec_0_lut = line_parameters.vector_0_lut;
    const auto* vec_1_lut = line_parameters.vector_1_lut;
    const auto& mu_y = line_parameters.mu_y;

    return (1 - mu_y) * ((1 - mu_x) * vec_0_lut[pixel_index] + mu_x * vec_0_lut[pixel_index + 1]) +
           mu_y * ((1 - mu_x) * vec_1_lut[pixel_index] + mu_x * vec_1_lut[pixel_index + 1]);
}

inline __device__ __host__ void AdjustDnImpl(double dn, double& calibration_value, double calibration_factor) {
    // TODO: think about the way to avoid this if-clause
    if (dn == snapengine::constants::THERMAL_NOISE_TRG_FLOOR_VALUE) {
        while (calibration_value < 0.00001) {
            dn *= 2;
            calibration_value = dn * calibration_factor;
        }
    }
}

inline __device__ __host__ double CalculateMuX(const CalibrationLineParameters& line_parameters, int x,
                                               int pixel_index) {
    auto mu_x = (x - line_parameters.calibration_vector_0->pixels[pixel_index]) /
                static_cast<double>(line_parameters.calibration_vector_0->pixels[pixel_index + 1] -
                                    line_parameters.calibration_vector_0->pixels[pixel_index]);
    if (isinf(mu_x)) {
        mu_x = (x - line_parameters.calibration_vector_0->pixels[pixel_index]) /
               static_cast<double>(line_parameters.calibration_vector_0->pixels[pixel_index + 1] -
                                   line_parameters.calibration_vector_0->pixels[pixel_index]);
    }

    return mu_x;
}

inline __device__ __host__ int CalculateSrcIndex(int x, int y, const Rectangle& target_rectangle) {
    return (x - target_rectangle.x) + (y - target_rectangle.y) * target_rectangle.width;
}

inline __device__ __host__ double CalculateCalibrationFactor(int x, const CalibrationLineParameters& line_parameters) {
    const auto pixel_index = GetPixelIndexImpl(x, line_parameters.calibration_vector_0);
    const auto mu_x = CalculateMuX(line_parameters, x, pixel_index);
    const auto lut_val = CalculateLutValImpl(line_parameters, mu_x, pixel_index);
    return 1.0 / (lut_val * lut_val);
}





inline __device__ __host__ void CalculateComplexIntensityImpl(int x, int y, CalibrationKernelArgs args, cuda::KernelArray<ComplexIntensityData> pixel_data) {
    const auto source_index = CalculateSrcIndex(x, y, args.target_rectangle);

    const double i = pixel_data.array[source_index].input.i;
    const double q = pixel_data.array[source_index].input.q;
    const double  dn = i * i + q * q;

    const auto calibration_factor =
        CalculateCalibrationFactor(x, args.line_parameters_array.array[y - args.target_rectangle.y]);
    double calibration_value = dn * calibration_factor;
    AdjustDnImpl(dn, calibration_value, calibration_factor);
    pixel_data.array[source_index].output = calibration_value;
}


}  // namespace sentinel1calibrate
}  // namespace alus