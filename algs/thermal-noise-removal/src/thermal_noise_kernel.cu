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
#include "thermal_noise_kernel.h"

#include <cstddef>
#include <vector>

#include <driver_types.h>

#include "cuda_manager.cuh"
#include "kernel_array.h"
#include "s1tbx-commons/noise_azimuth_vector.h"
#include "s1tbx-commons/noise_vector.h"
#include "thermal_noise_data_structures.h"

namespace alus::tnr {

/**
 * Interpolates noise azimuth vector values.
 * @param x_1 First line index.
 * @param x_2 Second line index.
 * @param y_1 First LUT value.
 * @param y_2 Second LUT value.
 * @param x Base line index.
 * @return Interpolated value.
 */
__device__ inline double Interpolate(int x_1, int x_2, double y_1, double y_2, int x) {
    if (x_1 ==
        x_2) {  // SNAP developers claim that this should never happen. // TODO: Consider removing this in order to
                //        avoid unnecessary branching
        return 0;
    }
    return y_1 + (static_cast<double>(x - x_1) / static_cast<double>(x_2 - x_1)) * (y_2 - y_1);
}

/**
 * Calculates index of the given range sample in the noise vector.
 *
 * @param sample Sample index of which is being enquired.
 * @param noise_vector Vector in which the sample index is being searched.
 * @return Index of the sample in a noise vector.
 */
__device__ inline size_t GetSampleIndex(int sample, s1tbx::DeviceNoiseVector noise_vector) {
    for (size_t i = 0; i < noise_vector.pixels.size; i++) {
        if (sample < noise_vector.pixels.array[i]) {
            return (i > 0) ? i - 1 : 0;
        }
    }

    return noise_vector.pixels.size - 2;
}

/**
 * Interpolates noise azimuth LUT on the line.
 *
 * @note This is a version for case when noise azimuth vector contains only one line.
 * @param noise_azimuth_vector Original azimuth vector.
 * @param first_azimuth_line Index of first azimuth line.
 * @param interpolated_vector Vector for storing interpolated values.
 * @param config Kernel launch configuration.
 */
__global__ void InterpolateNoiseAzimuthVectorSingleLine(s1tbx::DeviceNoiseAzimuthVector noise_azimuth_vector,
                                                        int first_azimuth_line,
                                                        cuda::KernelArray<double> interpolated_vector,
                                                        cuda::LaunchConfig1D config) {
    for (auto line : cuda::GpuGridRangeX(config.virtual_thread_count)) {
        interpolated_vector.array[line - first_azimuth_line] = noise_azimuth_vector.noise_azimuth_lut.array[0];
    }
}

/**
 * Interpolates noise azimuth LUT on the line.
 *
 * @note This is a version for case when noise azimuth vector contains more than one line.
 * @param noise_azimuth_vector Original azimuth vector.
 * @param first_azimuth_line Index of first azimuth line.
 * @param interpolated_vector Vector for storing interpolated values.
 * @param config Kernel launch configuration.
 */
__global__ void InterpolateNoiseAzimuthVectorMultiLine(s1tbx::DeviceNoiseAzimuthVector noise_azimuth_vector,
                                                       int first_azimuth_line, size_t line_index,
                                                       cuda::KernelArray<double> interpolated_vector,
                                                       cuda::LaunchConfig1D config) {
    for (auto line : cuda::GpuGridRangeX(config.virtual_thread_count)) {
        while (line_index < noise_azimuth_vector.lines.size - 2 &&
               line + first_azimuth_line > noise_azimuth_vector.lines.array[line_index + 1]) {
            line_index++;
        }

        auto* const lines = noise_azimuth_vector.lines.array;
        auto* const lut = noise_azimuth_vector.noise_azimuth_lut.array;

        interpolated_vector.array[line] = Interpolate(lines[line_index], lines[line_index + 1], lut[line_index],
                                                      lut[line_index + 1], line + first_azimuth_line);
    }
}

/**
 * Calculates indices of the first range samples in noise vectors.
 *
 * @param lines_per_burst Amount of lines in one burst.
 * @param first_range_sample First sample in the range.
 * @param burst_index_to_range_vector_map Map of range vectors.
 * @param launch_config Kernel launch configuration.
 */
__global__ void GetSampleIndexKernel(cuda::KernelArray<size_t> sample_indices, cuda::KernelArray<int> burst_indices,
                                     int first_range_sample,
                                     device::BurstIndexToRangeVectorMap burst_index_to_range_vector_map,
                                     cuda::LaunchConfig1D launch_config) {
    for (const auto index : cuda::GpuGridRangeX(launch_config.virtual_thread_count)) {
        const auto burst_index = burst_indices.array[index];
        const auto noise_range_vector = burst_index_to_range_vector_map.array[burst_index];
        const auto sample_index = GetSampleIndex(first_range_sample, noise_range_vector);
        sample_indices.array[index] = sample_index;
    }
}

/**
 * Interpolates Noise Range Vector unto bursts' row-values.
 *
 * @param burst_indices Array with selected bursts' indices.
 * @param sample_indices Array with sample indices.
 * @param first_range_sample Tile x0 value.
 * @param burst_index_to_range_vector_map Map with range vectors.
 * @param burst_index_to_interpolated_vector Map with interpolated vectors. These vectors' values will be overwritten by
 * this kernel.
 * @param launch_config Kernel launch parameters.
 */
__global__ void InterpolateNoiseRangeVectorKernel(
    cuda::KernelArray<int> burst_indices, cuda::KernelArray<size_t> sample_indices, int first_range_sample,
    device::BurstIndexToRangeVectorMap burst_index_to_range_vector_map,
    device::BurstIndexToInterpolatedRangeVectorMap burst_index_to_interpolated_vector,
    cuda::LaunchConfig2D launch_config) {
    for (const auto index : cuda::GpuGridRangeY(launch_config.virtual_thread_count.y)) {
        auto sample_index = sample_indices.array[index];
        const auto noise_range_vector = burst_index_to_range_vector_map.array[burst_indices.array[index]];
        auto interpolated_vector = burst_index_to_interpolated_vector.array[burst_indices.array[index]];
        for (const auto i : cuda::GpuGridRangeX(launch_config.virtual_thread_count.x)) {
            const auto sample = i + first_range_sample;
            while (sample_index < noise_range_vector.pixels.size - 2 &&
                   static_cast<int>(sample) > noise_range_vector.pixels.array[sample_index + 1]) {
                sample_index++;
            }
            interpolated_vector.array[i] = Interpolate(
                noise_range_vector.pixels.array[sample_index], noise_range_vector.pixels.array[sample_index + 1],
                noise_range_vector.noise_lut.array[sample_index], noise_range_vector.noise_lut.array[sample_index + 1],
                static_cast<int>(sample));
        }
    }
}

/**
 * Builds noise matrix for a tile using interpolated azimuth and range vectors.
 *
 * @param tile Target tile.
 * @param lines_per_burst Amount of lines in a single burst.
 * @param interpolated_azimuth_vector Interpolated azimuth vector.
 * @param range_vector_map Map binding burst index to interpolated range vectors.
 * @param noise_matrix Target noise matrix into which the data will be saved.
 * @param launch_config Kernel launch parameters.
 */
__global__ void CalculateNoiseMatrixKernelByBurst(Rectangle tile, int lines_per_burst,
                                                  cuda::KernelArray<double> interpolated_azimuth_vector,
                                                  device::BurstIndexToInterpolatedRangeVectorMap range_vector_map,
                                                  device::Matrix<double> noise_matrix,
                                                  cuda::LaunchConfig2D launch_config) {
    for (const auto y : cuda::GpuGridRangeY(launch_config.virtual_thread_count.y)) {
        const auto burst_index = (y + tile.y) / lines_per_burst;
        const auto& interpolated_range_vector = range_vector_map.array[burst_index];
        for (const auto x : cuda::GpuGridRangeX(launch_config.virtual_thread_count.x)) {
            noise_matrix.array[y].array[x] = interpolated_azimuth_vector.array[y] * interpolated_range_vector.array[x];
        }
    }
}

__global__ void ComputeComplexTileKernel(Rectangle tile, double no_data_value, double target_floor_data,
                                         device::Matrix<double> noise_matrix,
                                         cuda::KernelArray<IntensityData> pixel_data,
                                         cuda::LaunchConfig2D launch_config) {
    for (const auto y : cuda::GpuGridRangeY(launch_config.virtual_thread_count.y)) {
        for (const auto x : cuda::GpuGridRangeX(launch_config.virtual_thread_count.x)) {
            const auto pixel_index = y * tile.width + x;
            const double i = pixel_data.array[pixel_index].input_complex.i;
            const double q = pixel_data.array[pixel_index].input_complex.q;
            const auto dn_2 = i * i + q * q;
            double pixel_value;
            if (dn_2 == no_data_value) {
                pixel_value = no_data_value;
            } else {
                pixel_value = dn_2 - noise_matrix.array[y].array[x];
            }
            if (pixel_value < 0) {
                pixel_value = target_floor_data;
            }
            pixel_data.array[pixel_index].output = static_cast<float>(pixel_value);
        }
    }
}

__global__ void ComputeAmplitudeTileKernel(Rectangle tile, double no_data_value, double target_floor_data,
                                           device::Matrix<double> noise_matrix,
                                           cuda::KernelArray<IntensityData> pixel_data,
                                           cuda::LaunchConfig2D launch_config) {
    for (const auto y : cuda::GpuGridRangeY(launch_config.virtual_thread_count.y)) {
        for (const auto x : cuda::GpuGridRangeX(launch_config.virtual_thread_count.x)) {
            const auto pixel_index = y * tile.width + x;
            const double i = pixel_data.array[pixel_index].input_amplitude;
            const auto dn_2 = i * i;
            double pixel_value;
            if (dn_2 == no_data_value) {
                pixel_value = no_data_value;
            } else {
                pixel_value = dn_2 - noise_matrix.array[y].array[x];
            }
            if (pixel_value < 0) {
                pixel_value = target_floor_data;
            }
            pixel_data.array[pixel_index].output = static_cast<float>(pixel_value);
        }
    }
}

}  // namespace alus::tnr

alus::cuda::KernelArray<double> alus::tnr::LaunchInterpolateNoiseAzimuthVectorKernel(
    alus::s1tbx::DeviceNoiseAzimuthVector noise_azimuth_vector, int first_azimuth_line, int last_azimuth_line,
    size_t starting_line_index, cudaStream_t stream) {
    cuda::KernelArray<double> interpolated_vector{nullptr,
                                                  static_cast<size_t>(last_azimuth_line - first_azimuth_line + 1)};
    CHECK_CUDA_ERR(cudaMalloc(&interpolated_vector.array, interpolated_vector.ByteSize()));

    if (noise_azimuth_vector.lines.size < 2) {
        const auto launch_config = cuda::GetLaunchConfig1D(last_azimuth_line - first_azimuth_line + 1,
                                                           InterpolateNoiseAzimuthVectorSingleLine);
        InterpolateNoiseAzimuthVectorSingleLine<<<launch_config.block_count, launch_config.thread_per_block, 0,
                                                  stream>>>(noise_azimuth_vector, first_azimuth_line,
                                                            interpolated_vector, launch_config);
    } else {
        const auto launch_config =
            cuda::GetLaunchConfig1D(last_azimuth_line - first_azimuth_line + 1, InterpolateNoiseAzimuthVectorMultiLine);
        InterpolateNoiseAzimuthVectorMultiLine<<<launch_config.block_count, launch_config.thread_per_block, 0,
                                                 stream>>>(noise_azimuth_vector, first_azimuth_line,
                                                           starting_line_index, interpolated_vector, launch_config);
    }

    CHECK_CUDA_ERR(cudaDeviceSynchronize());
    CHECK_CUDA_ERR(cudaGetLastError());

    return interpolated_vector;
}

alus::tnr::device::BurstIndexToInterpolatedRangeVectorMap alus::tnr::LaunchInterpolateNoiseRangeVectorsKernel(
    alus::Rectangle tile, cuda::KernelArray<int> d_burst_indices, cuda::KernelArray<size_t> d_sample_indices,
    device::BurstIndexToRangeVectorMap burst_index_to_range_vector_map, cudaStream_t stream) {
    auto burst_index_to_interpolated_range_vector_map =
        device::CreateBurstIndexToInterpolatedRangeVectorMap(tile.width, burst_index_to_range_vector_map.size);

    const auto launch_config =
        cuda::GetLaunchConfig2D(tile.width, static_cast<int>(d_burst_indices.size), InterpolateNoiseRangeVectorKernel);

    InterpolateNoiseRangeVectorKernel<<<launch_config.grid_size, launch_config.block_size, 0, stream>>>(
        d_burst_indices, d_sample_indices, tile.x, burst_index_to_range_vector_map,
        burst_index_to_interpolated_range_vector_map, launch_config);

    CHECK_CUDA_ERR(cudaDeviceSynchronize());
    CHECK_CUDA_ERR(cudaGetLastError());
    return burst_index_to_interpolated_range_vector_map;
}

alus::cuda::KernelArray<size_t> alus::tnr::LaunchGetSampleIndexKernel(
    Rectangle tile, device::BurstIndexToRangeVectorMap burst_index_to_range_vector_map,
    cuda::KernelArray<int> burst_indices, cudaStream_t stream) {
    // Prepare sample indices array
    cuda::KernelArray<size_t> sample_indices_array{nullptr, burst_indices.size};
    CHECK_CUDA_ERR(cudaMalloc(&sample_indices_array.array, sample_indices_array.ByteSize()));

    const auto launch_config = cuda::GetLaunchConfig1D(static_cast<int>(burst_indices.size), GetSampleIndexKernel);
    GetSampleIndexKernel<<<launch_config.block_count, launch_config.thread_per_block, 0, stream>>>(
        sample_indices_array, burst_indices, tile.x, burst_index_to_range_vector_map, launch_config);

    return sample_indices_array;
}
alus::tnr::device::Matrix<double> alus::tnr::CalculateNoiseMatrix(
    alus::Rectangle tile, int lines_per_burst, alus::cuda::KernelArray<double> interpolated_azimuth_vector,
    alus::tnr::device::BurstIndexToInterpolatedRangeVectorMap range_vector_map, cudaStream_t stream) {
    auto d_matrix = device::CreateKernelMatrix<double>(tile.width, tile.height);

    const auto launch_config = cuda::GetLaunchConfig2D(tile.width, tile.height, CalculateNoiseMatrixKernelByBurst);

    CalculateNoiseMatrixKernelByBurst<<<launch_config.grid_size, launch_config.block_size, 0, stream>>>(
        tile, lines_per_burst, interpolated_azimuth_vector, range_vector_map, d_matrix, launch_config);

    return d_matrix;
}

void alus::tnr::LaunchComputeComplexTileKernel(alus::Rectangle tile, double no_data_value, double target_floor_value,
                                               cuda::KernelArray<IntensityData> pixel_data,
                                               device::Matrix<double> noise_matrix, cudaStream_t stream) {
    const auto launch_config = cuda::GetLaunchConfig2D(tile.width, tile.height, ComputeComplexTileKernel);
    ComputeComplexTileKernel<<<launch_config.grid_size, launch_config.block_size, 0, stream>>>(
        tile, no_data_value, target_floor_value, noise_matrix, pixel_data, launch_config);
}

void alus::tnr::LaunchComputeAmplitudeTileKernel(alus::Rectangle tile, double no_data_value, double target_floor_value,
                                                 cuda::KernelArray<IntensityData> pixel_data,
                                                 device::Matrix<double> noise_matrix, cudaStream_t stream) {
    const auto launch_config = cuda::GetLaunchConfig2D(tile.width, tile.height, ComputeAmplitudeTileKernel);
    ComputeAmplitudeTileKernel<<<launch_config.grid_size, launch_config.block_size, 0, stream>>>(
        tile, no_data_value, target_floor_value, noise_matrix, pixel_data, launch_config);
}
