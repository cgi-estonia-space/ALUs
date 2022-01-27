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

__global__ void CalculateComplexIntensityKernel(CalibrationKernelArgs args,
                                                cuda::KernelArray<ComplexIntensityData> pixel_data,
                                                cuda::LaunchConfig2D config) {
    for (auto x : cuda::GpuGridRangeX(config.virtual_thread_count.x)) {
        for (auto y : cuda::GpuGridRangeY(config.virtual_thread_count.y)) {
            const auto y_global = args.target_rectangle.y + y;
            const auto x_global = args.target_rectangle.x + x;

            CalculateComplexIntensityImpl(x_global, y_global, args, pixel_data);
        }
    }
}

__global__ void SetupTileLinesKernel(CalibrationKernelArgs args, cuda::LaunchConfig2D config) {
    for (auto y : cuda::GpuGridRangeY(config.virtual_thread_count.y)) {
        const auto y_global = static_cast<int>(y) + args.target_rectangle.y;

        SetupTileLineImpl(y_global, args, args.line_parameters_array.array[y]);
    }
}

void LaunchComplexIntensityKernel(CalibrationKernelArgs args, cuda::KernelArray<ComplexIntensityData> pixel_data,
                                  cudaStream_t stream) {
    const auto launch_config = cuda::GetLaunchConfig2D(args.target_rectangle.width, args.target_rectangle.height,
                                                       CalculateComplexIntensityKernel);
    CalculateComplexIntensityKernel<<<launch_config.grid_size, launch_config.block_size, 0, stream>>>(args, pixel_data,
                                                                                                      launch_config);

    // CHECK_CUDA_ERR(cudaStreamSynchronize(stream));
    // CHECK_CUDA_ERR(cudaGetLastError());
}

void LaunchSetupTileLinesKernel(CalibrationKernelArgs args, cudaStream_t stream) {
    const auto lines_count = args.target_rectangle.height;

    const auto launch_config = cuda::GetLaunchConfig2D(1, lines_count, SetupTileLinesKernel);
    SetupTileLinesKernel<<<launch_config.grid_size, launch_config.block_size, 0, stream>>>(args, launch_config);

    // CHECK_CUDA_ERR(cudaStreamSynchronize(stream));
    // CHECK_CUDA_ERR(cudaGetLastError());
}

void PopulateLUTs(cuda::KernelArray<CalibrationLineParameters> d_line_parameters, CAL_TYPE calibration_type,
                  CAL_TYPE data_type, cudaStream_t stream) {
    auto d_beg = thrust::device_pointer_cast(d_line_parameters.array);
    auto d_end = d_beg + d_line_parameters.size;
    switch (calibration_type) {
        case CAL_TYPE::SIGMA_0:
            thrust::transform(thrust::cuda::par.on(stream), d_beg, d_end, d_beg,
                              [=] __device__(CalibrationLineParameters line_parameters) {
                                  line_parameters.vector_0_lut = line_parameters.calibration_vector_0->sigma_nought;
                                  line_parameters.vector_1_lut = line_parameters.calibration_vector_1->sigma_nought;
                                  return line_parameters;
                              });
            break;
        case CAL_TYPE::BETA_0:
            thrust::transform(thrust::cuda::par.on(stream), d_beg, d_end, d_beg,
                              [] __device__(CalibrationLineParameters line_parameters) {
                                  line_parameters.vector_0_lut = line_parameters.calibration_vector_0->beta_nought;
                                  line_parameters.vector_1_lut = line_parameters.calibration_vector_1->beta_nought;
                                  return line_parameters;
                              });
            break;
        case CAL_TYPE::GAMMA:
            thrust::transform(thrust::cuda::par.on(stream), d_beg, d_end, d_beg,
                              [=] __device__(CalibrationLineParameters line_parameters) {
                                  line_parameters.vector_0_lut = line_parameters.calibration_vector_0->gamma;
                                  line_parameters.vector_1_lut = line_parameters.calibration_vector_1->gamma;
                                  return line_parameters;
                              });
            break;
        case CAL_TYPE::DN:
            thrust::transform(thrust::cuda::par.on(stream), d_beg, d_end, d_beg,
                              [=] __device__(CalibrationLineParameters line_parameters) {
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
                thrust::cuda::par.on(stream), d_beg, d_end, d_beg,
                [=] __device__(CalibrationLineParameters line_parameters) {
                    line_parameters.retro_vector_0_lut = line_parameters.calibration_vector_0->sigma_nought;
                    line_parameters.retor_vector_1_lut = line_parameters.calibration_vector_1->sigma_nought;
                    return line_parameters;
                });
            break;
        case CAL_TYPE::BETA_0:
            thrust::transform(
                thrust::cuda::par.on(stream), d_beg, d_end, d_beg,
                [=] __device__(CalibrationLineParameters line_parameters) {
                    line_parameters.retro_vector_0_lut = line_parameters.calibration_vector_0->beta_nought;
                    line_parameters.retor_vector_1_lut = line_parameters.calibration_vector_1->beta_nought;
                    return line_parameters;
                });
            break;
        case CAL_TYPE::GAMMA:
            thrust::transform(thrust::cuda::par.on(stream), d_beg, d_end, d_beg,
                              [=] __device__(CalibrationLineParameters line_parameters) {
                                  line_parameters.retro_vector_0_lut = line_parameters.calibration_vector_0->gamma;
                                  line_parameters.retor_vector_1_lut = line_parameters.calibration_vector_1->gamma;
                                  return line_parameters;
                              });
            break;
        case CAL_TYPE::DN:
            thrust::transform(thrust::cuda::par.on(stream), d_beg, d_end, d_beg,
                              [=] __device__(CalibrationLineParameters line_parameters) {
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
