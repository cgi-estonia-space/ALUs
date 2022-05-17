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
#include <vector>

#include <driver_types.h>

#include "kernel_array.h"
#include "s1tbx-commons/noise_azimuth_vector.h"
#include "s1tbx-commons/noise_vector.h"
#include "shapes.h"
#include "thermal_noise_data_structures.h"

namespace alus::tnr {

/**
 * Interpolates NoiseAzimuthVector on the given line.
 *
 * @param noise_azimuth_vector NoiseAzimuthVector acquired from the metadata.
 * @param first_azimuth_line
 * @param last_azimuth_line
 * @param starting_line_index Index, beginning from which interpolation will be performed.
 * @param stream CUDA stream on which the computation will be performed.
 * @return Interpolated noise azimuth vector.
 */
cuda::KernelArray<double> LaunchInterpolateNoiseAzimuthVectorKernel(
    s1tbx::DeviceNoiseAzimuthVector noise_azimuth_vector, int first_azimuth_line, int last_azimuth_line,
    size_t starting_line_index, cudaStream_t stream);

/**
 * Launches CUDA kernel that interpolates Noise Range Vectors' values unto bursts.
 *
 * @param tile Tile for which the calculation is performed.
 * @param d_burst_indices Device array with selected bursts' indices.
 * @param d_sample_indices Device array with sample indices.
 * @param burst_index_to_range_vector_map Device map with range vectors.
 * @param stream CUDA stream on which the computation will be performed.
 * @return Device map with interpolated vectors.
 */
device::BurstIndexToInterpolatedRangeVectorMap LaunchInterpolateNoiseRangeVectorsKernel(
    alus::Rectangle tile, cuda::KernelArray<int> d_burst_indices, cuda::KernelArray<size_t> d_sample_indices,
    device::BurstIndexToRangeVectorMap burst_index_to_range_vector_map, cudaStream_t stream);

/**
 * Gets sample indices for the given tile and range vectors.
 *
 * @param tile Tile dimensions.
 * @param burst_index_to_range_vector_map Kernel array acting as a map.
 * @param burst_indices
 * @result Kernel array with calculated sample indices.
 */
cuda::KernelArray<size_t> LaunchGetSampleIndexKernel(Rectangle tile,
                                                     device::BurstIndexToRangeVectorMap burst_index_to_range_vector_map,
                                                     cuda::KernelArray<int> burst_indices, cudaStream_t stream);

/**
 * Builds noise matrix for a tile using interpolated azimuth and range vectors.
 *
 * @param tile Target tile.
 * @param lines_per_burst Amount of lines in a single burst.
 * @param interpolated_azimuth_vector Interpolated azimuth vector.
 * @param range_vector_map Map binding burst index to interpolated range vectors.
 * @param stream CUDA stream on which the computation will be performed.
 * @note Azimuth and range noise vectors are expected to be allocated on the device.
 * @return Noise matrix allocated on the CUDA device.
 */
device::Matrix<double> CalculateNoiseMatrix(Rectangle tile, int lines_per_burst,
                                            cuda::KernelArray<double> interpolated_azimuth_vector,
                                            device::BurstIndexToInterpolatedRangeVectorMap range_vector_map,
                                            cudaStream_t stream);

/**
 * Performs the computation of input tile with the complex data.
 *
 * @param tile Input tile dimensions.
 * @param no_data_value No data value.
 * @param target_floor_value The minimal possible value.
 * @param pixel_data Input pixel values.
 * @param noise_matrix Noise lookup table.
 * @param stream CUDA stream on which the computation will be performed.
 */
void LaunchComputeComplexTileKernel(alus::Rectangle tile, double no_data_value, double target_floor_value,
                                    cuda::KernelArray<ComplexIntensityData> pixel_data,
                                    device::Matrix<double> noise_matrix, cudaStream_t stream);
}  // namespace alus::tnr