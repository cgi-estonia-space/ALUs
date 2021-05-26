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
#include "sentinel1_calibrate_kernel.h"

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "calibration_type.h"
#include "cuda_util.h"
#include "kernel_array.h"

#include "cuda_manager.cuh"
#include "sentinel1_calibrate_kernel_utils.cuh"

namespace alus {
namespace sentinel1calibrate {
__global__ void CalculateAmplitudeKernel(CalibrationKernelArgs args,
                                         cuda::KernelArray<CalibrationPixelParameters> pixel_parameters,
                                         cuda::KernelArray<double> calibration_values, cuda::LaunchConfig2D config) {
    for (auto x : cuda::GpuGridRangeX(config.virtual_thread_count.x)) {
        for (auto y : cuda::GpuGridRangeY(config.virtual_thread_count.y)) {
            const auto pixel_index = x + y * args.target_rectangle.width;
            CalculateAmplitudeImpl(pixel_parameters.array[pixel_index], calibration_values.array[pixel_index]);
        }
    }
}

__global__ void CalculateIntensityWithRetroKernel(CalibrationKernelArgs args,
                                                  cuda::KernelArray<CalibrationPixelParameters> pixel_parameters,
                                                  cuda::KernelArray<double> calibration_values,
                                                  cuda::LaunchConfig2D config) {
    for (auto x : cuda::GpuGridRangeX(config.virtual_thread_count.x)) {
        for (auto y : cuda::GpuGridRangeY(config.virtual_thread_count.y)) {
            const auto y_global = args.target_rectangle.y + y;

            const auto pixel_index = x + y * args.target_rectangle.width;
            CalculateIntensityWithRetroImpl(args.line_parameters_array.array[y_global], pixel_parameters.array[pixel_index],
                                        calibration_values.array[pixel_index]);
        }
    }
}

__global__ void CalculateIntensityWithoutRetroKernel(CalibrationKernelArgs args,
                                                     cuda::KernelArray<CalibrationPixelParameters> pixel_parameters,
                                                     cuda::KernelArray<double> calibration_values,
                                                     cuda::LaunchConfig2D config) {
    for (auto x : cuda::GpuGridRangeX(config.virtual_thread_count.x)) {
        for (auto y : cuda::GpuGridRangeY(config.virtual_thread_count.y)) {
            const auto pixel_index = x + y * args.target_rectangle.width;
            CalculateIntensityWithoutRetroImpl(pixel_parameters.array[pixel_index], calibration_values.array[pixel_index]);
        }
    }
}

__global__ void CalculateRealKernel(CalibrationKernelArgs args,
                                    cuda::KernelArray<CalibrationPixelParameters> pixel_parameters,
                                    cuda::KernelArray<double> calibration_values, cuda::LaunchConfig2D config) {
    for (auto x : cuda::GpuGridRangeX(config.virtual_thread_count.x)) {
        for (auto y : cuda::GpuGridRangeY(config.virtual_thread_count.y)) {
            const auto pixel_index = x + y * args.target_rectangle.width;
            CalculateRealImpl(args, pixel_parameters.array[pixel_index], calibration_values.array[pixel_index]);
        }
    }
}

__global__ void CalculateImaginaryKernel(CalibrationKernelArgs args,
                                         cuda::KernelArray<CalibrationPixelParameters> pixel_parameters,
                                         cuda::KernelArray<double> calibration_values, cuda::LaunchConfig2D config) {
    for (auto x : cuda::GpuGridRangeX(config.virtual_thread_count.x)) {
        for (auto y : cuda::GpuGridRangeY(config.virtual_thread_count.y)) {
            const auto pixel_index = x + y * args.target_rectangle.width;
            CalculateImaginaryImpl(args, pixel_parameters.array[pixel_index], calibration_values.array[pixel_index]);
        }
    }
}

__global__ void CalculateComplexIntensityKernel(CalibrationKernelArgs args,
                                                cuda::KernelArray<CalibrationPixelParameters> pixel_parameters,
                                                cuda::KernelArray<double> calibration_values,
                                                cuda::LaunchConfig2D config) {
    for (auto x : cuda::GpuGridRangeX(config.virtual_thread_count.x)) {
        for (auto y : cuda::GpuGridRangeY(config.virtual_thread_count.y)) {
            const auto pixel_index = x + y * args.target_rectangle.width;
            CalculateComplexIntensityImpl(args, pixel_parameters.array[pixel_index], calibration_values.array[pixel_index]);
        }
    }
}

__global__ void CalculateIntensityDBKernel(CalibrationKernelArgs args,
                                           cuda::KernelArray<CalibrationPixelParameters> pixel_parameters,
                                           cuda::KernelArray<double> calibration_values, cuda::LaunchConfig2D config) {
    for (auto x : cuda::GpuGridRangeX(config.virtual_thread_count.x)) {
        for (auto y : cuda::GpuGridRangeY(config.virtual_thread_count.y)) {
            const auto pixel_index = x + y * args.target_rectangle.width;
            CalculateIntensityDBImpl(pixel_parameters.array[pixel_index], calibration_values.array[pixel_index]);
        }
    }
}

__global__ void AdjustForComplexOutputKernel(CalibrationKernelArgs args,
                                             cuda::KernelArray<CalibrationPixelParameters> pixel_parameters,
                                             cuda::KernelArray<double> calibration_values,
                                             cuda::LaunchConfig2D config) {
    for (auto x : cuda::GpuGridRangeX(config.virtual_thread_count.x)) {
        for (auto y : cuda::GpuGridRangeY(config.virtual_thread_count.y)) {
            const auto pixel_index = x + y * args.target_rectangle.width;
            auto& calibration_value = calibration_values.array[pixel_index];

            calibration_value = std::sqrt(calibration_value) * pixel_parameters.array[pixel_index].phase_term;
        }
    }
}

__global__ void SetupTileLinesKernel(CalibrationKernelArgs args, cuda::LaunchConfig2D config) {
    for (auto y : cuda::GpuGridRangeY(config.virtual_thread_count.y)) {
        const auto y_global = static_cast<int>(y) + args.target_rectangle.y;

        SetupTileLineImpl(y_global, args, args.line_parameters_array.array[y]);
    }
}

__global__ void CalculatePixelParamsKernel(CalibrationKernelArgs args,
                                           cuda::KernelArray<CalibrationPixelParameters> pixel_parameters,
                                           cuda::LaunchConfig2D config) {
    for (auto x : cuda::GpuGridRangeX(config.virtual_thread_count.x)) {
        for (auto y : cuda::GpuGridRangeY(config.virtual_thread_count.y)) {
            const auto x_global = args.target_rectangle.x + x;
            const auto y_global = args.target_rectangle.y + y;

            CalculatePixelParamsImpl(static_cast<int>(x_global), static_cast<int>(y_global), args,
                                 args.line_parameters_array.array[y],
                                 pixel_parameters.array[x + y * args.target_rectangle.width]);
        }
    }
}

void LaunchSetupTileLinesKernel(CalibrationKernelArgs args) {
    const auto lines_count = args.target_rectangle.height;

    const auto launch_config = cuda::GetLaunchConfig2D(1, lines_count, SetupTileLinesKernel);
    SetupTileLinesKernel<<<launch_config.grid_size, launch_config.block_size>>>(args, launch_config);

    CHECK_CUDA_ERR(cudaDeviceSynchronize());

    CHECK_CUDA_ERR(cudaGetLastError());
}
void LaunchCalculatePixelParamsKernel(CalibrationKernelArgs args,
                                             cuda::KernelArray<CalibrationPixelParameters> pixel_parameters) {
    const auto launch_config =
        cuda::GetLaunchConfig2D(args.target_rectangle.width, args.target_rectangle.height, CalculatePixelParamsKernel);
    CalculatePixelParamsKernel<<<launch_config.grid_size, launch_config.block_size>>>(args, pixel_parameters,
                                                                                      launch_config);

    CHECK_CUDA_ERR(cudaDeviceSynchronize());

    CHECK_CUDA_ERR(cudaGetLastError());
}
void LaunchAmplitudeKernel(CalibrationKernelArgs args,
                                  cuda::KernelArray<CalibrationPixelParameters> pixel_parameters,
                                  cuda::KernelArray<double> calibration_values) {
    const auto launch_config =
        cuda::GetLaunchConfig2D(args.target_rectangle.width, args.target_rectangle.height, CalculateAmplitudeKernel);
    CalculateAmplitudeKernel<<<launch_config.grid_size, launch_config.block_size>>>(args, pixel_parameters,
                                                                                    calibration_values, launch_config);

    CHECK_CUDA_ERR(cudaDeviceSynchronize());

    CHECK_CUDA_ERR(cudaGetLastError());
}

void LaunchIntensityWithRetroKernel(CalibrationKernelArgs args,
                                           cuda::KernelArray<CalibrationPixelParameters> pixel_parameters,
                                           cuda::KernelArray<double> calibration_values) {
    const auto launch_config = cuda::GetLaunchConfig2D(args.target_rectangle.width, args.target_rectangle.height,
                                                       CalculateIntensityWithRetroKernel);
    CalculateIntensityWithRetroKernel<<<launch_config.grid_size, launch_config.block_size>>>(
        args, pixel_parameters, calibration_values, launch_config);

    CHECK_CUDA_ERR(cudaDeviceSynchronize());

    CHECK_CUDA_ERR(cudaGetLastError());
}

void LaunchIntensityWithoutRetroKernel(CalibrationKernelArgs args,
                                              cuda::KernelArray<CalibrationPixelParameters> pixel_parameters,
                                              cuda::KernelArray<double> calibration_values) {
    const auto launch_config = cuda::GetLaunchConfig2D(args.target_rectangle.width, args.target_rectangle.height,
                                                       CalculateIntensityWithoutRetroKernel);
    CalculateIntensityWithoutRetroKernel<<<launch_config.grid_size, launch_config.block_size>>>(
        args, pixel_parameters, calibration_values, launch_config);

    CHECK_CUDA_ERR(cudaDeviceSynchronize());

    CHECK_CUDA_ERR(cudaGetLastError());
}

void LaunchRealKernel(CalibrationKernelArgs args, cuda::KernelArray<CalibrationPixelParameters> pixel_parameters,
                             cuda::KernelArray<double> calibration_values) {
    const auto launch_config =
        cuda::GetLaunchConfig2D(args.target_rectangle.width, args.target_rectangle.height, CalculateRealKernel);
    CalculateRealKernel<<<launch_config.grid_size, launch_config.block_size>>>(args, pixel_parameters,
                                                                               calibration_values, launch_config);

    CHECK_CUDA_ERR(cudaDeviceSynchronize());

    CHECK_CUDA_ERR(cudaGetLastError());
}

void LaunchImaginaryKernel(CalibrationKernelArgs args,
                                  cuda::KernelArray<CalibrationPixelParameters> pixel_parameters,
                                  cuda::KernelArray<double> calibration_values) {
    const auto launch_config =
        cuda::GetLaunchConfig2D(args.target_rectangle.width, args.target_rectangle.height, CalculateImaginaryKernel);
    CalculateImaginaryKernel<<<launch_config.grid_size, launch_config.block_size>>>(args, pixel_parameters,
                                                                                    calibration_values, launch_config);

    CHECK_CUDA_ERR(cudaDeviceSynchronize());

    CHECK_CUDA_ERR(cudaGetLastError());
}

void LaunchComplexIntensityKernel(CalibrationKernelArgs args,
                                         cuda::KernelArray<CalibrationPixelParameters> pixel_parameters,
                                         cuda::KernelArray<double> calibration_values) {
    const auto launch_config =
        cuda::GetLaunchConfig2D(args.target_rectangle.width, args.target_rectangle.height, CalculateImaginaryKernel);
    CalculateComplexIntensityKernel<<<launch_config.grid_size, launch_config.block_size>>>(
        args, pixel_parameters, calibration_values, launch_config);

    CHECK_CUDA_ERR(cudaDeviceSynchronize());

    CHECK_CUDA_ERR(cudaGetLastError());
}

void LaunchIntensityDBKernel(CalibrationKernelArgs args,
                                    cuda::KernelArray<CalibrationPixelParameters> pixel_parameters,
                                    cuda::KernelArray<double> calibration_values) {
    const auto launch_config =
        cuda::GetLaunchConfig2D(args.target_rectangle.width, args.target_rectangle.height, CalculateIntensityDBKernel);
    CalculateIntensityDBKernel<<<launch_config.grid_size, launch_config.block_size>>>(
        args, pixel_parameters, calibration_values, launch_config);

    CHECK_CUDA_ERR(cudaDeviceSynchronize());

    CHECK_CUDA_ERR(cudaGetLastError());
}

void LaunchAdjustForCompexOutputKernel(CalibrationKernelArgs args,
                                              cuda::KernelArray<CalibrationPixelParameters> pixel_parameters,
                                              cuda::KernelArray<double> calibration_values) {
    const auto launch_config = cuda::GetLaunchConfig2D(args.target_rectangle.width, args.target_rectangle.height,
                                                       AdjustForComplexOutputKernel);
    AdjustForComplexOutputKernel<<<launch_config.grid_size, launch_config.block_size>>>(
        args, pixel_parameters, calibration_values, launch_config);

    CHECK_CUDA_ERR(cudaDeviceSynchronize());

    CHECK_CUDA_ERR(cudaGetLastError());
}

void PopulateLUTs(cuda::KernelArray<CalibrationLineParameters> d_line_parameters, CAL_TYPE calibration_type,
                  CAL_TYPE data_type) {
    switch (calibration_type) {
        case CAL_TYPE::SIGMA_0:
            thrust::transform(thrust::device, d_line_parameters.array, d_line_parameters.array + d_line_parameters.size,
                              d_line_parameters.array, [=] __device__(CalibrationLineParameters line_parameters) {
                                  line_parameters.vector_0_lut = line_parameters.calibration_vector_0->sigma_nought;
                                  line_parameters.vector_1_lut = line_parameters.calibration_vector_1->sigma_nought;
                                  return line_parameters;
                              });
            break;
        case CAL_TYPE::BETA_0:
            thrust::transform(thrust::device, d_line_parameters.array, d_line_parameters.array + d_line_parameters.size,
                              d_line_parameters.array, [] __device__(CalibrationLineParameters line_parameters) {
                                  line_parameters.vector_0_lut = line_parameters.calibration_vector_0->beta_nought;
                                  line_parameters.vector_1_lut = line_parameters.calibration_vector_1->beta_nought;
                                  return line_parameters;
                              });
            break;
        case CAL_TYPE::GAMMA:
            thrust::transform(thrust::device, d_line_parameters.array, d_line_parameters.array + d_line_parameters.size,
                              d_line_parameters.array, [=] __device__(CalibrationLineParameters line_parameters) {
                                  line_parameters.vector_0_lut = line_parameters.calibration_vector_0->gamma;
                                  line_parameters.vector_1_lut = line_parameters.calibration_vector_1->gamma;
                                  return line_parameters;
                              });
            break;
        case CAL_TYPE::DN:
            thrust::transform(thrust::device, d_line_parameters.array, d_line_parameters.array + d_line_parameters.size,
                              d_line_parameters.array, [=] __device__(CalibrationLineParameters line_parameters) {
                                  line_parameters.vector_0_lut = line_parameters.calibration_vector_0->dn;
                                  line_parameters.vector_1_lut = line_parameters.calibration_vector_1->dn;
                                  return line_parameters;
                              });
            break;
        case CAL_TYPE::NONE:
            break;
    }

    // Populate RetroLUTs
    switch (data_type) {
        case CAL_TYPE::SIGMA_0:
            thrust::transform(
                thrust::device, d_line_parameters.array, d_line_parameters.array + d_line_parameters.size,
                d_line_parameters.array, [=] __device__(CalibrationLineParameters line_parameters) {
                    line_parameters.retro_vector_0_lut = line_parameters.calibration_vector_0->sigma_nought;
                    line_parameters.retor_vector_1_lut = line_parameters.calibration_vector_1->sigma_nought;
                    return line_parameters;
                });
            break;
        case CAL_TYPE::BETA_0:
            thrust::transform(
                thrust::device, d_line_parameters.array, d_line_parameters.array + d_line_parameters.size,
                d_line_parameters.array, [=] __device__(CalibrationLineParameters line_parameters) {
                    line_parameters.retro_vector_0_lut = line_parameters.calibration_vector_0->beta_nought;
                    line_parameters.retor_vector_1_lut = line_parameters.calibration_vector_1->beta_nought;
                    return line_parameters;
                });
            break;
        case CAL_TYPE::GAMMA:
            thrust::transform(thrust::device, d_line_parameters.array, d_line_parameters.array + d_line_parameters.size,
                              d_line_parameters.array, [=] __device__(CalibrationLineParameters line_parameters) {
                                  line_parameters.retro_vector_0_lut = line_parameters.calibration_vector_0->gamma;
                                  line_parameters.retor_vector_1_lut = line_parameters.calibration_vector_1->gamma;
                                  return line_parameters;
                              });
            break;
        case CAL_TYPE::DN:
            thrust::transform(thrust::device, d_line_parameters.array, d_line_parameters.array + d_line_parameters.size,
                              d_line_parameters.array, [=] __device__(CalibrationLineParameters line_parameters) {
                                  line_parameters.retro_vector_0_lut = line_parameters.calibration_vector_0->dn;
                                  line_parameters.retor_vector_1_lut = line_parameters.calibration_vector_1->dn;
                                  return line_parameters;
                              });
            break;
        case CAL_TYPE::NONE:
            break;
    }
}
}  // namespace sentinel1calibrate
}  // namespace alus
