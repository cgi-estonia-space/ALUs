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

#include "nppi_resample.h"

#include <cuda_runtime.h>
#include <nppi_geometry_transforms.h>

#include "cuda_util.h"
#include "raster_properties.h"
#include "resample_method.h"

namespace {
NppiInterpolationMode GetInterpolation(alus::resample::Method method) {
    switch (method) {
        case alus::resample::Method::SMOOTH_EDGE:
            return NppiInterpolationMode::NPPI_SMOOTH_EDGE;
        case alus::resample::Method::LANCZOS3:
            return NppiInterpolationMode::NPPI_INTER_LANCZOS3_ADVANCED;
        case alus::resample::Method::LANCZOS:
            return NppiInterpolationMode::NPPI_INTER_LANCZOS;
        case alus::resample::Method::CUBIC2P_C05C03:
            return NppiInterpolationMode::NPPI_INTER_CUBIC2P_B05C03;
        case alus::resample::Method::CUBIC2P_CATMULLROM:
            return NppiInterpolationMode::NPPI_INTER_CUBIC2P_CATMULLROM;
        case alus::resample::Method::CUBIC2P_BSPLINE:
            return NppiInterpolationMode::NPPI_INTER_CUBIC2P_BSPLINE;
        case alus::resample::Method::LINEAR:
            return NppiInterpolationMode::NPPI_INTER_LINEAR;
        case alus::resample::Method::NEAREST_NEIGHBOUR:
            return NppiInterpolationMode::NPPI_INTER_NN;
        case alus::resample::Method::SUPER:
            return NppiInterpolationMode::NPPI_INTER_SUPER;
        default:
            return NppiInterpolationMode::NPPI_INTER_UNDEFINED;
    }
}
}  // namespace

namespace alus::resample {

void PrepareDeviceBuffers(const void* input_data, void** device_mem_input, size_t input_size, void** device_mem_output,
                          size_t output_size) {
    CHECK_CUDA_ERR(cudaMalloc(device_mem_input, input_size));
    CHECK_CUDA_ERR(cudaMemcpy(*device_mem_input, input_data, input_size, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMalloc(device_mem_output, output_size));
}

void UnloadDeviceBuffers(void* device_mem_input, void* device_mem_output) {
    CHECK_CUDA_ERR(cudaFree(device_mem_input));
    CHECK_CUDA_ERR(cudaFree(device_mem_output));
}

void UnloadDeviceBuffers(void* device_mem_input, void* device_mem_output, void* output_data, size_t output_size) {
    CHECK_CUDA_ERR(cudaMemcpy(output_data, device_mem_output, output_size, cudaMemcpyDeviceToHost));
    UnloadDeviceBuffers(device_mem_input, device_mem_output);
}

int DoResampling(const void* input, RasterDimension input_dimension, void* output, RasterDimension output_dimension,
                 TypeParameters type_parameter, Method method) {
    CHECK_CUDA_ERR(cudaDeviceSynchronize());
    const auto is_floating_point{type_parameter.is_float};
    const auto pixel_type_size{type_parameter.size_bytes};
    const auto is_signed{type_parameter.is_signed};

    NppStatus status = NppStatus::NPP_DATA_TYPE_ERROR;
    if (is_floating_point) {
        if (pixel_type_size == 4) {
            status = nppiResize_32f_C1R(
                static_cast<const Npp32f*>(input), input_dimension.columnsX * pixel_type_size,
                {input_dimension.columnsX, input_dimension.rowsY},
                {0, 0, input_dimension.columnsX, input_dimension.rowsY}, static_cast<Npp32f*>(output),
                output_dimension.columnsX * pixel_type_size, {output_dimension.columnsX, output_dimension.rowsY},
                {0, 0, output_dimension.columnsX, output_dimension.rowsY}, GetInterpolation(method));
        }
    } else if (pixel_type_size == 1) {
        if (!is_signed) {
            status = nppiResize_8u_C1R(
                static_cast<const Npp8u*>(input), input_dimension.columnsX * pixel_type_size,
                {input_dimension.columnsX, input_dimension.rowsY},
                {0, 0, input_dimension.columnsX, input_dimension.rowsY}, static_cast<Npp8u*>(output),
                output_dimension.columnsX * pixel_type_size, {output_dimension.columnsX, output_dimension.rowsY},
                {0, 0, output_dimension.columnsX, output_dimension.rowsY}, GetInterpolation(method));
        }
    } else if (pixel_type_size == 2) {
        if (is_signed) {
            status = nppiResize_16s_C1R(
                static_cast<const Npp16s*>(input), input_dimension.columnsX * pixel_type_size,
                {input_dimension.columnsX, input_dimension.rowsY},
                {0, 0, input_dimension.columnsX, input_dimension.rowsY}, static_cast<Npp16s*>(output),
                output_dimension.columnsX * pixel_type_size, {output_dimension.columnsX, output_dimension.rowsY},
                {0, 0, output_dimension.columnsX, output_dimension.rowsY}, GetInterpolation(method));

        } else {
            status = nppiResize_16u_C1R(
                static_cast<const Npp16u*>(input), input_dimension.columnsX * pixel_type_size,
                {input_dimension.columnsX, input_dimension.rowsY},
                {0, 0, input_dimension.columnsX, input_dimension.rowsY}, static_cast<Npp16u*>(output),
                output_dimension.columnsX * pixel_type_size, {output_dimension.columnsX, output_dimension.rowsY},
                {0, 0, output_dimension.columnsX, output_dimension.rowsY}, GetInterpolation(method));
        }
    }

    CHECK_CUDA_ERR(cudaDeviceSynchronize());

    return status;
}

}  // namespace alus::resample