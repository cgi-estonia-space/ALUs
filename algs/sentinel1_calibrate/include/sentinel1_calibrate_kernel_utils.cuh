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

    return mathutils::ChooseOne(index == snapengine::constants::INVALID_INDEX, static_cast<int64_t>(snapengine::constants::INVALID_INDEX),
                                index - 1);
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

inline __device__ __host__ double CalculateLutValImpl(CalibrationLineParameters& line_parameters,
                                                      CalibrationPixelParameters& pixel_parameters) {
    const auto& mu_x = pixel_parameters.mu_x;
    const auto& mu_y = line_parameters.mu_y;
    const auto& pixel_index = pixel_parameters.pixel_index;
    const auto* vec_0_lut = line_parameters.vector_0_lut;
    const auto* vec_1_lut = line_parameters.vector_1_lut;

    return (1 - mu_y) * ((1 - mu_x) * vec_0_lut[pixel_index] + mu_x * vec_0_lut[pixel_index + 1]) +
           mu_y * ((1 - mu_x) * vec_1_lut[pixel_index] + mu_x * vec_1_lut[pixel_index + 1]);
}

inline __device__ __host__ void CalculatePixelParamsImpl(int x, int y, CalibrationKernelArgs& args,
                                                         CalibrationLineParameters& line_parameters,
                                                         CalibrationPixelParameters& pixel_parameters) {
    const auto pixel_index =
        (x - args.target_rectangle.x) + (y - args.target_rectangle.y) * args.target_rectangle.width;
    pixel_parameters.source_index = pixel_index;
    pixel_parameters.dn = args.source_data_1.array[pixel_index];
    pixel_parameters.pixel_index = GetPixelIndexImpl(x, line_parameters.calibration_vector_0);
    pixel_parameters.mu_x =
        (x - line_parameters.calibration_vector_0->pixels[pixel_parameters.pixel_index]) /
        static_cast<double>(line_parameters.calibration_vector_0->pixels[pixel_parameters.pixel_index + 1] -
                            line_parameters.calibration_vector_0->pixels[pixel_parameters.pixel_index]);
    if (isinf(pixel_parameters.mu_x)) {
        pixel_parameters.mu_x =
            (x - line_parameters.calibration_vector_0->pixels[pixel_parameters.pixel_index]) /
            static_cast<double>(line_parameters.calibration_vector_0->pixels[pixel_parameters.pixel_index + 1] -
                                line_parameters.calibration_vector_0->pixels[pixel_parameters.pixel_index]);
    }
    pixel_parameters.lut_val = CalculateLutValImpl(line_parameters, pixel_parameters);
    pixel_parameters.lut_val = CalculateLutValImpl(line_parameters, pixel_parameters);
    pixel_parameters.calibration_factor = 1.0 / (pixel_parameters.lut_val * pixel_parameters.lut_val);
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

inline __device__ __host__ void CalculateAmplitudeImpl(CalibrationPixelParameters& parameters,
                                                       double& calibration_value) {
    parameters.dn *= parameters.dn;
    calibration_value = parameters.dn * parameters.calibration_factor;
    AdjustDnImpl(parameters.dn, calibration_value, parameters.calibration_factor);
}

inline __device__ __host__ void CalculateIntensityWithRetroImpl(CalibrationLineParameters& line_parameters,
                                                                CalibrationPixelParameters& pixel_parameters,
                                                                double& calibration_value) {
    const auto& mu_x = pixel_parameters.mu_x;
    const auto& mu_y = line_parameters.mu_y;
    const auto& pixel_index = pixel_parameters.pixel_index;
    auto* const retro_vec_0_lut = line_parameters.retro_vector_0_lut;
    auto* const retro_vec_1_lut = line_parameters.retor_vector_1_lut;

    const auto retro_lut_val =
        (1 - mu_y) * ((1 - mu_x) * retro_vec_0_lut[pixel_index] + mu_x * retro_vec_0_lut[pixel_index + 1]) +
        mu_y * ((1 - mu_x) * retro_vec_1_lut[pixel_index] + mu_x * retro_vec_1_lut[pixel_index + 1]);

    pixel_parameters.calibration_factor *= retro_lut_val;
    calibration_value = pixel_parameters.dn * pixel_parameters.calibration_factor;
    AdjustDnImpl(pixel_parameters.dn, calibration_value, pixel_parameters.calibration_factor);
}

inline __device__ __host__ void CalculateIntensityWithoutRetroImpl(CalibrationPixelParameters& pixel_parameters,
                                                                   double& calibration_value) {
    pixel_parameters.calibration_factor *= pixel_parameters.retro_lut_val;
    calibration_value = pixel_parameters.dn * pixel_parameters.calibration_factor;
    AdjustDnImpl(pixel_parameters.dn, calibration_value, pixel_parameters.calibration_factor);
}

inline __device__ __host__ void CalculateRealImpl(CalibrationKernelArgs args, CalibrationPixelParameters& parameters,
                                                  double& calibration_value) {
    const auto i = parameters.dn;
    const auto q = args.source_data_2.array[parameters.source_index];
    parameters.dn = i * i + q * q;
    if (parameters.dn > 0.0) {
        parameters.phase_term = i / std::sqrt(parameters.dn);
    } else {
        parameters.phase_term = 0.0;
    }
    calibration_value = parameters.dn * parameters.calibration_factor;
    AdjustDnImpl(parameters.dn, calibration_value, parameters.calibration_factor);
}

inline __device__ __host__ void CalculateImaginaryImpl(CalibrationKernelArgs args,
                                                       CalibrationPixelParameters& parameters,
                                                       double& calibration_value) {
    const auto i = parameters.dn;
    const auto q = args.source_data_2.array[parameters.source_index];
    parameters.dn = i * i + q * q;
    if (parameters.dn > 0.0) {
        parameters.phase_term = q / std::sqrt(parameters.dn);
    } else {
        parameters.phase_term = 0;
    }
    calibration_value = parameters.dn * parameters.calibration_factor;
    AdjustDnImpl(parameters.dn, calibration_value, parameters.calibration_factor);
}

inline __device__ __host__ void CalculateComplexIntensityImpl(CalibrationKernelArgs args,
                                                              CalibrationPixelParameters& parameters,
                                                              double& calibration_value) {
    const auto i = parameters.dn;
    const auto q = args.source_data_2.array[parameters.source_index];
    parameters.dn = i * i + q * q;
    parameters.phase_term = 0;

    calibration_value = parameters.dn * parameters.calibration_factor;
    AdjustDnImpl(parameters.dn, calibration_value, parameters.calibration_factor);
}

inline __device__ __host__ void CalculateIntensityDBImpl(CalibrationPixelParameters& parameters,
                                                         double& calibration_value) {
    parameters.dn = std::pow(10, parameters.dn / 10.0);
    calibration_value = parameters.dn * parameters.calibration_factor;
    AdjustDnImpl(parameters.dn, calibration_value, parameters.calibration_factor);
}
}  // namespace sentinel1calibrate
}  // namespace alus