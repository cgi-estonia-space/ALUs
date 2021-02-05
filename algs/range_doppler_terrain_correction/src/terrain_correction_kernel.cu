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
#include "range_doppler_geocoding.cuh"
#include "raster_utils.cuh"
#include "sar_geocoding.cuh"
#include "srtm3_elevation_calc.cuh"

#include "cuda_util.hpp"
#include "position_data.h"
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

__global__ void TerrainCorrectionKernel(TcTileCoordinates tile_coordinates, cuda::KernelArray<double> target,
                                        TerrainCorrectionKernelArgs args) {
    auto const thread_x = threadIdx.x + blockIdx.x * blockDim.x;
    auto const thread_y = threadIdx.y + blockIdx.y * blockDim.y;

    if (thread_x >= tile_coordinates.target_width || thread_y >= tile_coordinates.target_height) {
        return;
    }

    auto const target_tile_index = thread_y * tile_coordinates.target_width + thread_x;

    const PrecisePixelPosition pixel_position{thread_x + tile_coordinates.target_x_0,
                                              thread_y + tile_coordinates.target_y_0};

    auto temp_coordinates = rasterutils::CalculatePixelCoordinates(pixel_position, args.target_geo_transform);

    const Coordinates coordinates{temp_coordinates.lon < 180.0 ? temp_coordinates.lon : temp_coordinates.lon - 360.0,
                                  temp_coordinates.lat};

    PointerArray srtm_3_tiles_array{};
    srtm_3_tiles_array.array = const_cast<PointerHolder*>(args.srtm_3_tiles);
    double const altitude = args.use_avg_scene_height ? args.avg_scene_height
                                                      : snapengine::srtm3elevationmodel::GetElevation(
                                                            coordinates.lat, coordinates.lon, &srtm_3_tiles_array);

    if (altitude == args.dem_no_data_value) {
        SetNoDataValue(target_tile_index, args.target_no_data_value, target.array);
        return;
    }

    s1tbx::PositionData position_data{};
    if (!GetPositionImpl(coordinates.lat, coordinates.lon, altitude, position_data, args.get_position_metadata)) {
        SetNoDataValue(target_tile_index, args.target_no_data_value, target.array);
        return;
    }

    if (!s1tbx::sargeocoding::IsValidCellImpl(position_data.range_index, position_data.azimuth_index, args.diff_lat,
                                              args.source_image_width - 1, args.source_image_height - 1)) {
        SetNoDataValue(target_tile_index, args.target_no_data_value, target.array);
        return;
    }

    double index_i[2];
    double index_j[2];
    double index_ki[1];
    double index_kj[1];

    args.resampling_index = {0, 0, 0, 0, 0, 0, index_i, index_j, index_ki, index_kj};
    int sub_swath_index = INVALID_SUB_SWATH_INDEX;

    double v = rangedopplergeocoding::GetPixelValue(position_data.azimuth_index, position_data.range_index, 1,
                                                    args.source_image_width, args.source_image_height,
                                                    args.resampling_raster, args.resampling_index, sub_swath_index);
    target.array[target_tile_index] = v;
}

cudaError_t LaunchTerrainCorrectionKernel(TcTile tile, TerrainCorrectionKernelArgs args) {
    const int block_dim = 20;  // TODO: this number should somehow be calculated. Block thread count should ideally be
                               //   divisible by 32. (SNAPGPU-163)
    dim3 block_size{block_dim, block_dim};
    dim3 main_kernel_grid_size{static_cast<unsigned int>(tile.tc_tile_coordinates.target_width / block_size.x + 1),
                               static_cast<unsigned int>(tile.tc_tile_coordinates.target_height / block_size.y + 1)};

    thrust::device_vector<double> d_target(tile.target_tile_data_buffer.size);
    cuda::KernelArray<double> kernel_target{thrust::raw_pointer_cast(d_target.data()), d_target.size()};

    if (args.resampling_raster.source_rectangle_calculated) {
        double* d_source_tile_data{};
        const auto source_tile_data_size =
            tile.tc_tile_coordinates.source_height * tile.tc_tile_coordinates.source_width * sizeof(double);
        CHECK_CUDA_ERR(cudaMalloc(&d_source_tile_data, source_tile_data_size));
        CHECK_CUDA_ERR(cudaMemcpy(d_source_tile_data, args.resampling_raster.source_tile_i->data_buffer,
                                  source_tile_data_size, cudaMemcpyHostToDevice));
        args.resampling_raster.source_tile_i->data_buffer = d_source_tile_data;

        snapengine::resampling::Tile* d_source_tile;
        CHECK_CUDA_ERR(cudaMalloc(&d_source_tile, sizeof(snapengine::resampling::Tile)));
        CHECK_CUDA_ERR(cudaMemcpy(d_source_tile, args.resampling_raster.source_tile_i,
                                  sizeof(snapengine::resampling::Tile), cudaMemcpyHostToDevice));
        args.resampling_raster.source_tile_i = d_source_tile;
    }

    TerrainCorrectionKernel<<<main_kernel_grid_size, block_size>>>(tile.tc_tile_coordinates, kernel_target, args);

    cudaDeviceSynchronize();

    cudaError_t error = cudaGetLastError();

    CHECK_CUDA_ERR(cudaMemcpy(tile.target_tile_data_buffer.array, kernel_target.array,
                              sizeof(double) * kernel_target.size, cudaMemcpyDeviceToHost));

    return error;
}

__global__ void GetSourceRectangleKernel(TcTileCoordinates tile_coordinates,
                                         GeoTransformParameters target_geo_transform, const PointerHolder* srtm_3_tiles,
                                         double dem_no_data_value, GetPositionMetadata get_position_metadata,
                                         size_t num_points_per_row, size_t num_points_per_col, int* x_max, int* x_min,
                                         int* y_max, int* y_min, double avg_scene_height, bool use_avg_scene_height) {
    const auto index_col = threadIdx.x + blockIdx.x * blockDim.x;
    const auto index_row = threadIdx.y + blockIdx.y * blockDim.y;
    if (index_col >= num_points_per_row || index_row >= num_points_per_col) {
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

    Coordinates coordinates{
        target_geo_transform.originLon + x * target_geo_transform.pixelSizeLon + target_geo_transform.pixelSizeLon / 2,
        target_geo_transform.originLat + y * target_geo_transform.pixelSizeLat + target_geo_transform.pixelSizeLat / 2};

    PointerArray srtm_3_tiles_array{};
    srtm_3_tiles_array.array = const_cast<PointerHolder*>(srtm_3_tiles);
    const auto altitude = use_avg_scene_height ? avg_scene_height
                                               : snapengine::srtm3elevationmodel::GetElevation(
                                                     coordinates.lat, coordinates.lon, &srtm_3_tiles_array);

    if (altitude == dem_no_data_value) {
        return;
    }
    s1tbx::PositionData position_data{};
    if (!GetPositionImpl(coordinates.lat, coordinates.lon, altitude, position_data, get_position_metadata)) {
        return;
    }

    const auto index = index_row * num_points_per_row + index_col;
    x_max[index] = std::ceil(position_data.range_index);
    x_min[index] = std::floor(position_data.range_index);
    y_max[index] = std::ceil(position_data.azimuth_index);
    y_min[index] = std::floor(position_data.azimuth_index);
}

bool GetSourceRectangle(TcTile& tile, GeoTransformParameters target_geo_transform, double dem_no_data_value,
                        size_t source_image_width, size_t source_image_height, GetPositionMetadata get_position_metadata,
                        Rectangle& source_rectangle, const PointerHolder* srtm_3_tiles, double avg_scene_height,
                        bool use_avg_height) {
    const int num_points_per_row = 5;
    const int num_points_per_col = 5;

    thrust::device_vector<int> x_max_values(num_points_per_row * num_points_per_col, INT_MIN);
    thrust::device_vector<int> x_min_values(num_points_per_row * num_points_per_col, INT_MAX);
    thrust::device_vector<int> y_max_values(num_points_per_row * num_points_per_col, INT_MIN);
    thrust::device_vector<int> y_min_values(num_points_per_row * num_points_per_col, INT_MAX);

    const dim3 block_dim{num_points_per_row, num_points_per_col};
    const dim3 grid_dim{1};
    GetSourceRectangleKernel<<<grid_dim, block_dim>>>(
        tile.tc_tile_coordinates, target_geo_transform, srtm_3_tiles, dem_no_data_value, get_position_metadata,
        num_points_per_row, num_points_per_col, thrust::raw_pointer_cast(x_max_values.data()),
        thrust::raw_pointer_cast(x_min_values.data()), thrust::raw_pointer_cast(y_max_values.data()),
        thrust::raw_pointer_cast(y_min_values.data()), avg_scene_height, use_avg_height);

    CHECK_CUDA_ERR(cudaGetLastError());
    CHECK_CUDA_ERR(cudaDeviceSynchronize());
    const auto margin = BILINEAR_INTERPOLATION_MARGIN;

    const auto x_min =
        std::max(*thrust::min_element(thrust::device, x_min_values.begin(), x_min_values.end()) - margin, 0);
    const auto x_max = std::min(*thrust::max_element(thrust::device, x_max_values.begin(), x_max_values.end()) + margin,
                                static_cast<int>(source_image_width - 1));
    const auto y_min =
        std::max(*thrust::min_element(thrust::device, y_min_values.begin(), y_min_values.end()) - margin, 0);
    const auto y_max = std::min(*thrust::max_element(thrust::device, y_max_values.begin(), y_max_values.end()) + margin,
                                static_cast<int>(source_image_height - 1));

    if (x_min > x_max || y_min > y_max) {
        return false;
    }

    source_rectangle = {x_min, y_min, x_max - x_min + 1, y_max - y_min + 1};

    return true;
}
}  // namespace terraincorrection
}  // namespace alus