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
#include <cmath>
#include <cstddef>

#include <thrust/device_vector.h>

#include "get_position.cuh"
#include "math_utils.cuh"
#include "range_doppler_geocoding.cuh"
#include "raster_utils.cuh"
#include "sar_geocoding.cuh"
#include "srtm3_elevation_calc.cuh"

#include "cuda_util.hpp"
#include "position_data.h"
#include "raster_properties.hpp"
#include "tc_tile.h"
#include "terrain_correction_constants.h"
#include "terrain_correction_kernel.h"

namespace alus {
namespace terraincorrection {
__global__ void CalculateVelocitiesAndPositionsKernel(
    const double first_line_utc, const double line_time_interval,
    cuda::KernelArray<snapengine::OrbitStateVectorComputation> vectors,
    cuda::KernelArray<snapengine::PosVector> velocities, cuda::KernelArray<snapengine::PosVector> positions) {
    const auto block_id = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
    const auto index = block_id * (blockDim.x * blockDim.y * blockDim.z) + (threadIdx.z * (blockDim.x * blockDim.y)) +
                       (threadIdx.y * blockDim.x) + threadIdx.x;
    if (index >= velocities.size) {
        return;
    }

    const double time = first_line_utc + index * line_time_interval;
    const double dt =
        (vectors.array[vectors.size - 1].timeMjd_ - vectors.array[0].timeMjd_) / static_cast<double>(vectors.size - 1);

    s1tbx::orbitstatevectors::GetPositionVelocity(time, vectors.array, vectors.size, dt, &positions.array[index],
                                                  &velocities.array[index]);
}

void CalculateVelocitiesAndPositions(const int source_image_height, const double first_line_utc,
                                     const double line_time_interval,
                                     cuda::KernelArray<snapengine::OrbitStateVectorComputation> vectors,
                                     cuda::KernelArray<snapengine::PosVector> velocities,
                                     cuda::KernelArray<snapengine::PosVector> positions) {
    dim3 block_dim{1024};
    dim3 grid_dim{static_cast<unsigned int>(cuda::GetGridDim(block_dim.x, source_image_height))};

    CalculateVelocitiesAndPositionsKernel<<<grid_dim, block_dim>>>(first_line_utc, line_time_interval, vectors,
                                                                   velocities, positions);

    CHECK_CUDA_ERR(cudaGetLastError());
    CHECK_CUDA_ERR(cudaDeviceSynchronize());
};

__inline__ __device__ void SetNoDataValue(int index, double value, double* dest_data) { dest_data[index] = value; }

__device__ Coordinates GetPixelCoordinates(TcTileCoordinates tile_coordinates,
                                           GeoTransformParameters target_geo_transform, unsigned int thread_x,
                                           unsigned int thread_y) {
    const PrecisePixelPosition pixel_position{thread_x + tile_coordinates.target_x_0,
                                              thread_y + tile_coordinates.target_y_0};
    const auto temp_coordinates = rasterutils::CalculatePixelCoordinates(pixel_position, target_geo_transform);
    return {mathutils::ChooseOne(temp_coordinates.lon < 180.0, temp_coordinates.lon, temp_coordinates.lon - 360.0),
            temp_coordinates.lat};
}

__device__ bool CheckPositionAndCellValidity(s1tbx::PositionData& position_data, Coordinates coordinates,
                                             double altitude, TerrainCorrectionKernelArgs args) {
    if (!GetPositionImpl(coordinates.lat, coordinates.lon, altitude, position_data, args.get_position_metadata)) {
        return false;
    }

    if (!s1tbx::sargeocoding::IsValidCellImpl(position_data.range_index, position_data.azimuth_index, args.diff_lat,
                                              args.source_image_width - 1, args.source_image_height - 1)) {
        return false;
    }

    return true;
}

__global__ void TerrainCorrectionKernel(TcTileCoordinates tile_coordinates, cuda::KernelArray<double> target,
                                        TerrainCorrectionKernelArgs args) {
    const auto thread_x = threadIdx.x + blockIdx.x * blockDim.x;
    const auto thread_y = threadIdx.y + blockIdx.y * blockDim.y;

    if (thread_x >= tile_coordinates.target_width || thread_y >= tile_coordinates.target_height) {
        return;
    }

    const auto index = thread_y * tile_coordinates.target_width + thread_x;

    if (args.valid_pixels.array[index] == false) {
        SetNoDataValue(index, args.target_no_data_value, target.array);
        return;
    }

    const auto coordinates = GetPixelCoordinates(tile_coordinates, args.target_geo_transform, thread_x, thread_y);

    const auto altitude = snapengine::srtm3elevationmodel::GetElevation(coordinates.lat, coordinates.lon,
                                                                        const_cast<PointerArray*>(&args.srtm_3_tiles));

    if (altitude == args.dem_no_data_value) {
        SetNoDataValue(index, args.target_no_data_value, target.array);
        return;
    }

    s1tbx::PositionData position_data{};
    if (!CheckPositionAndCellValidity(position_data, coordinates, altitude, args)) {
        SetNoDataValue(index, args.target_no_data_value, target.array);
        return;
    }

    double index_i[2];
    double index_j[2];
    double index_ki[1];
    double index_kj[1];

    args.resampling_index = {0, 0, 0, 0, 0, 0, index_i, index_j, index_ki, index_kj};
    int sub_swath_index = INVALID_SUB_SWATH_INDEX;

    double v = rangedopplergeocoding::GetPixelValue(
        position_data.azimuth_index, position_data.range_index, BILINEAR_INTERPOLATION_MARGIN, args.source_image_width,
        args.source_image_height, args.resampling_raster, args.resampling_index, sub_swath_index);
    target.array[index] = v;
}

__global__ void TerrainCorrectionWithAverageHeightKernel(TcTileCoordinates tile_coordinates,
                                                         cuda::KernelArray<double> target,
                                                         TerrainCorrectionKernelArgs args) {
    auto const thread_x = threadIdx.x + blockIdx.x * blockDim.x;
    auto const thread_y = threadIdx.y + blockIdx.y * blockDim.y;

    if (thread_x >= tile_coordinates.target_width || thread_y >= tile_coordinates.target_height) {
        return;
    }

    auto const pixel_index = thread_y * tile_coordinates.target_width + thread_x;

    if (args.valid_pixels.array[pixel_index] == false) {
        SetNoDataValue(pixel_index, args.target_no_data_value, target.array);
        return;
    }

    const auto coordinates = GetPixelCoordinates(tile_coordinates, args.target_geo_transform, thread_x, thread_y);

    s1tbx::PositionData position_data{};
    if (!CheckPositionAndCellValidity(position_data, coordinates, args.avg_scene_height, args)) {
        SetNoDataValue(pixel_index, args.target_no_data_value, target.array);
        return;
    }

    double index_i[2];
    double index_j[2];
    double index_ki[1];
    double index_kj[1];

    args.resampling_index = {0, 0, 0, 0, 0, 0, index_i, index_j, index_ki, index_kj};
    int sub_swath_index = INVALID_SUB_SWATH_INDEX;

    double v = rangedopplergeocoding::GetPixelValue(
        position_data.azimuth_index, position_data.range_index, BILINEAR_INTERPOLATION_MARGIN, args.source_image_width,
        args.source_image_height, args.resampling_raster, args.resampling_index, sub_swath_index);
    target.array[pixel_index] = v;
}

cudaError_t LaunchTerrainCorrectionKernel(TcTile tile, TerrainCorrectionKernelArgs args) {
    const int block_dim = 20;  // TODO: this number should somehow be calculated. Block thread count should ideally be
                               //   divisible by 32. (SNAPGPU-216) (SNAPGPU-211 should introduce mechanism for
                               //   calculating prefect kernel dimensions)
    dim3 block_size{block_dim, block_dim};
    dim3 main_kernel_grid_size{static_cast<unsigned int>(tile.tc_tile_coordinates.target_width / block_size.x + 1),
                               static_cast<unsigned int>(tile.tc_tile_coordinates.target_height / block_size.y + 1)};

    thrust::device_vector<double> d_target(tile.target_tile_data_buffer.size);
    cuda::KernelArray<double> kernel_target{thrust::raw_pointer_cast(d_target.data()), d_target.size()};

    snapengine::resampling::Tile* d_source_tile;
    CHECK_CUDA_ERR(cudaMalloc(&d_source_tile, sizeof(snapengine::resampling::Tile)));
    CHECK_CUDA_ERR(cudaMemcpy(d_source_tile, args.resampling_raster.source_tile_i, sizeof(snapengine::resampling::Tile),
                              cudaMemcpyHostToDevice));
    args.resampling_raster.source_tile_i = d_source_tile;

    if (args.use_avg_scene_height) {
        TerrainCorrectionWithAverageHeightKernel<<<main_kernel_grid_size, block_size>>>(tile.tc_tile_coordinates,
                                                                                        kernel_target, args);
    } else {
        TerrainCorrectionKernel<<<main_kernel_grid_size, block_size>>>(tile.tc_tile_coordinates, kernel_target, args);
    }

    cudaDeviceSynchronize();

    cudaError_t error = cudaGetLastError();

    CHECK_CUDA_ERR(cudaMemcpy(tile.target_tile_data_buffer.array, kernel_target.array,
                              sizeof(double) * kernel_target.size, cudaMemcpyDeviceToHost));

    return error;
}

__global__ void GetNonBorderSourceRectangleKernel(TcTileCoordinates tile_coordinates, TerrainCorrectionKernelArgs args,
                                                  int num_points_per_row, int num_points_per_col, int* x_max,
                                                  int* x_min, int* y_max, int* y_min) {
    const auto index_col = threadIdx.x + blockIdx.x * blockDim.x;
    const auto index_row = threadIdx.y + blockIdx.y * blockDim.y;
    if (index_col >= static_cast<unsigned int>(num_points_per_row) ||
        index_row >= static_cast<unsigned int>(num_points_per_col)) {
        return;
    }

    const auto x_offset = tile_coordinates.target_width / (num_points_per_row - 1);
    const auto y_offset = tile_coordinates.target_height / (num_points_per_col - 1);

    auto y = index_row == static_cast<size_t>(num_points_per_col - 1)
                 ? tile_coordinates.target_y_0 + tile_coordinates.target_height - 1
                 : tile_coordinates.target_y_0 + static_cast<double>(index_row) * y_offset;
    auto x = index_col == static_cast<size_t>(num_points_per_row - 1)
                 ? tile_coordinates.target_x_0 + tile_coordinates.target_width - 1
                 : tile_coordinates.target_x_0 + static_cast<double>(index_col) * x_offset;

    Coordinates coordinates = rasterutils::CalculatePixelCoordinates({x, y}, args.target_geo_transform);

    // TODO: can be split into two kernels in order to avoid branching (SNAPGPU-216)
    const auto altitude = args.use_avg_scene_height
                              ? args.avg_scene_height
                              : snapengine::srtm3elevationmodel::GetElevation(
                                    coordinates.lat, coordinates.lon, const_cast<PointerArray*>(&args.srtm_3_tiles));

    if (altitude == args.dem_no_data_value) {
        return;
    }
    s1tbx::PositionData position_data{};
    if (!GetPositionImpl(coordinates.lat, coordinates.lon, altitude, position_data, args.get_position_metadata)) {
        return;
    }

    const auto index = index_row * num_points_per_row + index_col;
    x_max[index] = std::ceil(position_data.range_index);
    x_min[index] = std::floor(position_data.range_index);
    y_max[index] = std::ceil(position_data.azimuth_index);
    y_min[index] = std::floor(position_data.azimuth_index);
}

bool GetNonBorderSourceRectangle(TcTile& tile, TerrainCorrectionKernelArgs args,
                                 Rectangle& source_rectangle) {
    const int num_points_per_row = 5;
    const int num_points_per_col = 5;

    const size_t vector_length = num_points_per_row * num_points_per_col;

    thrust::device_vector<int> x_max_values(vector_length, INT_MIN);
    thrust::device_vector<int> x_min_values(vector_length, INT_MAX);
    thrust::device_vector<int> y_max_values(vector_length, INT_MIN);
    thrust::device_vector<int> y_min_values(vector_length, INT_MAX);

    const dim3 block_dim{num_points_per_row, num_points_per_col};
    const dim3 grid_dim{1};
    GetNonBorderSourceRectangleKernel<<<grid_dim, block_dim>>>(
        tile.tc_tile_coordinates, args, num_points_per_row, num_points_per_col,
        thrust::raw_pointer_cast(x_max_values.data()), thrust::raw_pointer_cast(x_min_values.data()),
        thrust::raw_pointer_cast(y_max_values.data()), thrust::raw_pointer_cast(y_min_values.data()));

    CHECK_CUDA_ERR(cudaGetLastError());
    CHECK_CUDA_ERR(cudaDeviceSynchronize());
    const auto margin = BILINEAR_INTERPOLATION_MARGIN;

    const auto x_min =
        std::max(*thrust::min_element(thrust::device, x_min_values.begin(), x_min_values.end()) - margin, 0);
    const auto x_max = std::min(*thrust::max_element(thrust::device, x_max_values.begin(), x_max_values.end()) + margin,
                                static_cast<int>(args.source_image_width - 1));
    const auto y_min =
        std::max(*thrust::min_element(thrust::device, y_min_values.begin(), y_min_values.end()) - margin, 0);
    const auto y_max = std::min(*thrust::max_element(thrust::device, y_max_values.begin(), y_max_values.end()) + margin,
                                static_cast<int>(args.source_image_height - 1));

    if (x_min > x_max || y_min > y_max) {
        return false;
    }

    source_rectangle = {x_min, y_min, x_max - x_min + 1, y_max - y_min + 1};

    return true;
}

__global__ void GetSourceRectangleWithAverageHeightKernel(TcTileCoordinates tile_coordinates,
                                                          TerrainCorrectionKernelArgs args, int* range_indices,
                                                          int* azimuth_indices) {
    const auto thread_x = threadIdx.x + blockIdx.x * blockDim.x;
    const auto thread_y = threadIdx.y + blockIdx.y * blockDim.y;

    if (thread_x >= tile_coordinates.target_width || thread_y >= tile_coordinates.target_height) {
        return;
    }

    const auto index = thread_y * tile_coordinates.target_width + thread_x;

    args.valid_pixels.array[index] = false;

    const auto coordinates = GetPixelCoordinates(tile_coordinates, args.target_geo_transform, thread_x, thread_y);

    s1tbx::PositionData position_data{};
    if (!CheckPositionAndCellValidity(position_data, coordinates, args.avg_scene_height, args)) {
        return;
    }

    args.valid_pixels.array[index] = true;

    range_indices[index] = static_cast<int>(position_data.range_index);
    azimuth_indices[index] = static_cast<int>(position_data.azimuth_index);
}

__global__ void GetSourceRectangleKernel(TcTileCoordinates tile_coordinates, TerrainCorrectionKernelArgs args,
                                         int* range_indices, int* azimuth_indices) {
    const auto thread_x = threadIdx.x + blockIdx.x * blockDim.x;
    const auto thread_y = threadIdx.y + blockIdx.y * blockDim.y;

    if (thread_x >= tile_coordinates.target_width || thread_y >= tile_coordinates.target_height) {
        return;
    }

    const auto index = thread_y * tile_coordinates.target_width + thread_x;
    args.valid_pixels.array[index] = false;

    const auto coordinates = GetPixelCoordinates(tile_coordinates, args.target_geo_transform, thread_x, thread_y);

    const auto altitude = snapengine::srtm3elevationmodel::GetElevation(coordinates.lat, coordinates.lon,
                                                                        const_cast<PointerArray*>(&args.srtm_3_tiles));

    if (altitude == args.dem_no_data_value) {
        return;
    }

    s1tbx::PositionData position_data{};
    if (!CheckPositionAndCellValidity(position_data, coordinates, altitude, args)) {
        return;
    }

    args.valid_pixels.array[index] = true;

    range_indices[index] = static_cast<int>(position_data.range_index);
    azimuth_indices[index] = static_cast<int>(position_data.azimuth_index);
}

Rectangle GetSourceRectangle(TcTileCoordinates tile_coordinates, TerrainCorrectionKernelArgs args) {
    thrust::device_vector<int> range_indices(tile_coordinates.target_width * tile_coordinates.target_height, INT_MIN);
    thrust::device_vector<int> azimuth_indices(tile_coordinates.target_width * tile_coordinates.target_height, INT_MIN);

    dim3 block_dim{32, 32};
    dim3 grid_dim{static_cast<uint>(cuda::GetGridDim(block_dim.x, tile_coordinates.target_width)),
                  static_cast<uint>(cuda::GetGridDim(block_dim.y, tile_coordinates.target_height))};

    if (args.use_avg_scene_height) {
        GetSourceRectangleWithAverageHeightKernel<<<grid_dim, block_dim>>>(
            tile_coordinates, args, thrust::raw_pointer_cast(range_indices.data()),
            thrust::raw_pointer_cast(azimuth_indices.data()));
    } else {
        GetSourceRectangleKernel<<<grid_dim, block_dim>>>(tile_coordinates, args,
                                                          thrust::raw_pointer_cast(range_indices.data()),
                                                          thrust::raw_pointer_cast(azimuth_indices.data()));
    }

    CHECK_CUDA_ERR(cudaGetLastError());
    CHECK_CUDA_ERR(cudaDeviceSynchronize());

    const auto margin = BILINEAR_INTERPOLATION_MARGIN;
    thrust::device_vector<int> temp(tile_coordinates.target_width * tile_coordinates.target_height);

    auto temp_last_element = thrust::copy_if(thrust::device, range_indices.begin(), range_indices.end(), temp.begin(),
                                             [] __device__(int range_index) { return range_index != INT_MIN; });
    const auto range_minmax = thrust::minmax_element(thrust::device, temp.begin(), temp.end());

    const auto x_min = std::max(0, *range_minmax.first - margin);
    const auto x_max = std::min(static_cast<int>(args.source_image_width), *range_minmax.second + 2 * margin + 1);

    temp.clear();
    temp_last_element = thrust::copy_if(thrust::device, azimuth_indices.begin(), azimuth_indices.end(), temp.begin(),
                                        [] __device__(int azimuth_index) { return azimuth_index != INT_MIN; });
    const auto azimuth_minmax = thrust::minmax_element(thrust::device, temp.begin(), temp_last_element);

    const auto y_min = std::max(0, *azimuth_minmax.first - margin);
    const auto y_max = std::min(static_cast<int>(args.source_image_height), *azimuth_minmax.second + 2 * margin + 1);

    return {x_min, y_min, x_max - x_min, y_max - y_min};
}

}  // namespace terraincorrection
}  // namespace alus
