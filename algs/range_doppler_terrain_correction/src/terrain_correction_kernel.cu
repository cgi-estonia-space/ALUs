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
#include <thrust/device_vector.h>

#include "position_data.h"
#include "tc_tile.h"

#include "terrain_correction_kernel.h"
#include "terrain_correction_constants.h"

#include "get_position.cuh"
#include "local_dem.cuh"
#include "range_doppler_geocoding.cuh"
#include "raster_utils.cuh"
#include "sar_geocoding.cuh"

namespace alus {
namespace terraincorrection {
__global__ void CalculateVelocitiesAndPositionsKernel(
    const double first_line_utc,
    const double line_time_interval,
    cuda::KernelArray<snapengine::OrbitStateVectorComputation> vectors,
    cuda::KernelArray<snapengine::PosVector> velocities,
    cuda::KernelArray<snapengine::PosVector> positions) {
    const auto block_id = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
    const auto index = block_id * (blockDim.x * blockDim.y * blockDim.z) + (threadIdx.z * (blockDim.x * blockDim.y)) +
                       (threadIdx.y * blockDim.x) + threadIdx.x;
    if (index >= velocities.size) {
        return;
    }

    const double time = first_line_utc + index * line_time_interval;
    const double dt =
        (vectors.array[vectors.size - 1].timeMjd_ - vectors.array[0].timeMjd_) / static_cast<double>(vectors.size - 1);

    s1tbx::orbitstatevectors::GetPositionVelocity(
        time, vectors.array, vectors.size, dt, &positions.array[index], &velocities.array[index]);
}

void CalculateVelocitiesAndPositions(const int source_image_height,
                                     const double first_line_utc,
                                     const double line_time_interval,
                                     cuda::KernelArray<snapengine::OrbitStateVectorComputation> vectors,
                                     cuda::KernelArray<snapengine::PosVector> velocities,
                                     cuda::KernelArray<snapengine::PosVector> positions) {
    dim3 block_dim{32, 32};
    dim3 grid_dim{static_cast<unsigned int>(source_image_height / (32 * 32) + 1)};

    CalculateVelocitiesAndPositionsKernel<<<grid_dim, block_dim>>>(
        first_line_utc, line_time_interval, vectors, velocities, positions);

    CHECK_CUDA_ERR(cudaGetLastError());
    CHECK_CUDA_ERR(cudaDeviceSynchronize());
};

__device__ void SetNoDataValue(int index, double value, double *dest_data) { dest_data[index] = value; }

__global__ void TerrainCorrectionKernel(TcTileCoordinates tile_coordinates,
                                        double avg_scene_height,
                                        cuda::KernelArray<double> target,
                                        TerrainCorrectionKernelArgs args) {
    auto const thread_x = threadIdx.x + blockIdx.x * blockDim.x;
    auto const thread_y = threadIdx.y + blockIdx.y * blockDim.y;

    if (thread_x >= tile_coordinates.target_width || thread_y >= tile_coordinates.target_height) {
        return;
    }

    auto const target_tile_index = thread_y * tile_coordinates.target_width + thread_x;

    double const altitude = avg_scene_height;

    if (altitude == args.dem_no_data_value) {
        SetNoDataValue(target_tile_index, args.target_no_data_value, target.array);
        return;
    }

    const PrecisePixelPosition pixel_position{thread_x + tile_coordinates.target_x_0,
                                              thread_y + tile_coordinates.target_y_0};

    auto temp_coordinates = rasterutils::CalculatePixelCoordinates(pixel_position, args.target_geo_transform);

    const Coordinates coordinates{temp_coordinates.lon < 180.0 ? temp_coordinates.lon : temp_coordinates.lon - 360.0,
                                  temp_coordinates.lat};

    s1tbx::PositionData position_data{};
    if (!GetPositionImpl(coordinates.lat, coordinates.lon, altitude, position_data, args.get_position_metadata)) {
        SetNoDataValue(target_tile_index, 0.0, target.array);
        return;
    }

    if (!s1tbx::sargeocoding::IsValidCellImpl(position_data.range_index,
                                              position_data.azimuth_index,
                                              args.diff_lat,
                                              args.source_image_width - 1,
                                              args.source_image_height - 1)) {
        SetNoDataValue(target_tile_index, 0.0, target.array);
        return;
    }

    double index_i[2];
    double index_j[2];
    double index_ki[1];
    double index_kj[1];

    args.resampling_index = {0, 0, 0, 0, 0, 0, index_i, index_j, index_ki, index_kj};
    int sub_swath_index = INVALID_SUB_SWATH_INDEX;

    double v = rangedopplergeocoding::GetPixelValue(position_data.azimuth_index,
                                                    position_data.range_index,
                                                    1,
                                                    args.source_image_width,
                                                    args.source_image_height,
                                                    args.resampling_raster,
                                                    args.resampling_index,
                                                    sub_swath_index);
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
        double *d_source_tile_data{};
        const auto source_tile_data_size =
            tile.tc_tile_coordinates.source_height * tile.tc_tile_coordinates.source_width * sizeof(double);
        CHECK_CUDA_ERR(cudaMalloc(&d_source_tile_data, source_tile_data_size));
        CHECK_CUDA_ERR(cudaMemcpy(d_source_tile_data,
                                  args.resampling_raster.source_tile_i->data_buffer,
                                  source_tile_data_size,
                                  cudaMemcpyHostToDevice));
        args.resampling_raster.source_tile_i->data_buffer = d_source_tile_data;

        snapengine::resampling::Tile *d_source_tile;
        CHECK_CUDA_ERR(cudaMalloc(&d_source_tile, sizeof(snapengine::resampling::Tile)));
        CHECK_CUDA_ERR(cudaMemcpy(d_source_tile,
                                  args.resampling_raster.source_tile_i,
                                  sizeof(snapengine::resampling::Tile),
                                  cudaMemcpyHostToDevice));
        args.resampling_raster.source_tile_i = d_source_tile;
    }

    TerrainCorrectionKernel<<<main_kernel_grid_size, block_size>>>(
        tile.tc_tile_coordinates, args.avg_scene_height, kernel_target, args);

    cudaDeviceSynchronize();

    cudaError_t error = cudaGetLastError();

    CHECK_CUDA_ERR(cudaMemcpy(tile.target_tile_data_buffer.array,
                              kernel_target.array,
                              sizeof(double) * kernel_target.size,
                              cudaMemcpyDeviceToHost));

    return error;
}

bool GetSourceRectangle(TcTile &tile,
                        GeoTransformParameters target_geo_transform,
                        double dem_no_data_value,
                        double avg_scene_height,
                        int source_image_width,
                        int source_image_height,
                        GetPositionMetadata get_position_metadata,
                        Rectangle &source_rectangle) {
    const int num_points_per_row = 5;
    const int num_points_per_col = 5;
    const int x_offset = tile.tc_tile_coordinates.target_width / (num_points_per_row - 1);
    const int y_offset = tile.tc_tile_coordinates.target_height / (num_points_per_col - 1);
    const int margin = BILINEAR_INTERPOLATION_MARGIN;  // Specific value for Bilinear Interpolation

    int x_max = INT_MIN;
    int x_min = INT_MAX;
    int y_max = INT_MIN;
    int y_min = INT_MAX;

    for (int i = 0; i < num_points_per_col; i++) {
        const double Y = i == num_points_per_col - 1
                             ? tile.tc_tile_coordinates.target_y_0 + tile.tc_tile_coordinates.target_height - 1
                             : tile.tc_tile_coordinates.target_y_0 + i * y_offset;

        for (int j = 0; j < num_points_per_row; j++) {
            const double x = j == num_points_per_row - 1
                                 ? tile.tc_tile_coordinates.target_x_0 + tile.tc_tile_coordinates.target_width - 1
                                 : tile.tc_tile_coordinates.target_x_0 + j * x_offset;

            double altitude = avg_scene_height;  // TODO: implement SRTM3 tiles (SNAPGPU-191, SNAPGPU-193)
            if (altitude == dem_no_data_value) {
                continue;
            }

            Coordinates coordinates{target_geo_transform.originLon + x * target_geo_transform.pixelSizeLon +
                                        target_geo_transform.pixelSizeLon / 2,
                                    target_geo_transform.originLat + Y * target_geo_transform.pixelSizeLat +
                                        target_geo_transform.pixelSizeLat / 2};

            s1tbx::PositionData position_data{};

            if (!GetPositionImpl(coordinates.lat, coordinates.lon, altitude, position_data, get_position_metadata)) {
                continue;
            }

            if (x_max < position_data.range_index) {
                x_max = std::ceil(position_data.range_index);
            }

            if (x_min > position_data.range_index) {
                x_min = std::floor(position_data.range_index);
            }

            if (y_max < position_data.azimuth_index) {
                y_max = std::ceil(position_data.azimuth_index);
            }

            if (y_min > position_data.azimuth_index) {
                y_min = std::floor(position_data.azimuth_index);
            }
        }
    }

    x_min = std::max(x_min - margin, 0);
    x_max = std::min(x_max + margin, source_image_width - 1);
    y_min = std::max(y_min - margin, 0);
    y_max = std::min(y_max + margin, source_image_height - 1);

    if (x_min > x_max || y_min > y_max) {
        return false;
    }

    source_rectangle = {x_min, y_min, x_max - x_min + 1, y_max - y_min + 1};

    return true;
}
}  // namespace terraincorrection
}  // namespace alus