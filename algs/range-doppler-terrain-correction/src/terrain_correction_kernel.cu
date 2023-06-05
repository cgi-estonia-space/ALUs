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

#include "copdem_cog_30m_calc.cuh"
#include "dem_calc.cuh"
#include "get_position.cuh"
#include "math_utils.cuh"
#include "range_doppler_geocoding.cuh"
#include "raster_utils.cuh"

#include "calc_kernels.cuh"
#include "tc_tile.h"
#include "terrain_correction_constants.h"
#include "terrain_correction_kernel.h"

namespace alus::terraincorrection {
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

    auto pos_vel = s1tbx::orbitstatevectors::GetPositionVelocity(time, vectors.array, vectors.size, dt);
    positions.array[index] = pos_vel.position;
    velocities.array[index] = pos_vel.velocity;
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

template <typename T>
__inline__ __device__ void SetNoDataValue(int index, T value, T* dest_data) {
    dest_data[index] = value;
}

__device__ Coordinates GetPixelCoordinates(TcTileIndexPair tile_coordinates,
                                           GeoTransformParameters target_geo_transform, unsigned int thread_x,
                                           unsigned int thread_y) {
    const PrecisePixelPosition pixel_position{static_cast<double>(thread_x + tile_coordinates.target_x_0),
                                              static_cast<double>(thread_y + tile_coordinates.target_y_0)};
    const auto temp_coordinates = rasterutils::CalculatePixelCoordinates(pixel_position, target_geo_transform);
    return {mathutils::ChooseOne(temp_coordinates.lon < 180.0, temp_coordinates.lon, temp_coordinates.lon - 360.0),
            temp_coordinates.lat};
}

__device__ bool CheckPositionAndCellValidity(s1tbx::PositionData& position_data, Coordinates coordinates,
                                             double altitude, GetSourceRectangleKernelArgs args) {
    if (!GetPositionImpl(coordinates.lat, coordinates.lon, altitude, position_data, args.get_position_metadata,
                         args.d_srgr_coefficients)) {
        return false;
    }

    if (!s1tbx::sargeocoding::IsValidCellImpl(position_data.range_index, position_data.azimuth_index, args.diff_lat,
                                              args.source_image_width - 1, args.source_image_height - 1)) {
        return false;
    }

    return true;
}

__global__ void TerrainCorrectionKernel(TcTileIndexPair tile_coordinates, cuda::KernelArray<float> target,
                                        TerrainCorrectionKernelArgs args,
                                        snapengine::resampling::ResamplingRaster resampling_raster) {
    const auto thread_x = threadIdx.x + blockIdx.x * blockDim.x;
    const auto thread_y = threadIdx.y + blockIdx.y * blockDim.y;

    if (thread_x >= tile_coordinates.target_width || thread_y >= tile_coordinates.target_height) {
        return;
    }

    const auto index = thread_y * tile_coordinates.target_width + thread_x;

    const double azimuth_index = args.d_azimuth_index[index];
    if (std::isnan(azimuth_index)) {
        SetNoDataValue(index, static_cast<float>(args.target_no_data_value), target.array);
        return;
    }

    double index_i[2];
    double index_j[2];
    double index_ki[1];
    double index_kj[1];

    snapengine::resampling::ResamplingIndex resampling_index = {0, 0, 0, 0, 0, 0, index_i, index_j, index_ki, index_kj};
    int sub_swath_index = INVALID_SUB_SWATH_INDEX;

    double v = rangedopplergeocoding::GetPixelValue(
        azimuth_index, args.d_range_index[index], BILINEAR_INTERPOLATION_MARGIN, args.source_image_width,
        args.source_image_height, resampling_raster, resampling_index, sub_swath_index);
    target.array[index] = v;
}

cudaError_t LaunchTerrainCorrectionKernel(TcTileIndexPair tc_tile_coordinates, TerrainCorrectionKernelArgs args,
                                          float* h_target_buffer, cudaStream_t stream) {
    dim3 block_size{16, 16};
    dim3 main_kernel_grid_size{static_cast<unsigned int>(tc_tile_coordinates.target_width / block_size.x + 1),
                               static_cast<unsigned int>(tc_tile_coordinates.target_height / block_size.y + 1)};

    TerrainCorrectionKernel<<<main_kernel_grid_size, block_size, 0, stream>>>(tc_tile_coordinates, args.d_target, args,
                                                                              args.resampling_raster);
    if (args.db_values) {
        math::calckernels::CalcDb<<<main_kernel_grid_size, block_size, 0, stream>>>(
            args.d_target, tc_tile_coordinates.target_width, tc_tile_coordinates.target_height,
            args.target_no_data_value);
    }

    cuda::CopyArrayAsyncD2H(h_target_buffer, args.d_target.array, args.d_target.size, stream);

    // need to wait for async memcpy to complete
    CHECK_CUDA_ERR(cudaStreamSynchronize(stream));

    return cudaGetLastError();
}

__global__ void GetSourceRectangleKernel(TcTileIndexPair tile_coordinates, GetSourceRectangleKernelArgs args,
                                         SourceRectangeResult* result) {
    const auto thread_x = threadIdx.x + blockIdx.x * blockDim.x;
    const auto thread_y = threadIdx.y + blockIdx.y * blockDim.y;

    if (thread_x >= tile_coordinates.target_width || thread_y >= tile_coordinates.target_height) {
        return;
    }

    const auto index = thread_y * tile_coordinates.target_width + thread_x;

    const auto coordinates = GetPixelCoordinates(tile_coordinates, args.target_geo_transform, thread_x, thread_y);

    double altitude = 0;
    if (args.use_avg_scene_height) {
        altitude = args.avg_scene_height;
    } else if (args.dem_type == dem::Type::COPDEM_COG30m) {
        altitude = dem::CopDemCog30mGetElevation(coordinates.lat, coordinates.lon, &args.dem_tiles, args.dem_property);
        if (altitude == args.dem_no_data_value) {
            args.d_azimuth_index[index] = CUDART_NAN;
            return;
        }
    } else if (args.dem_type == dem::Type::SRTM3) {
        altitude = snapengine::dem::GetElevation(coordinates.lat, coordinates.lon, &args.dem_tiles, args.dem_property);
        if (altitude == args.dem_no_data_value) {
            args.d_azimuth_index[index] = CUDART_NAN;
            return;
        }
    }

    s1tbx::PositionData position_data{};
    if (!CheckPositionAndCellValidity(position_data, coordinates, altitude, args)) {
        args.d_azimuth_index[index] = CUDART_NAN;
        return;
    }
    args.d_azimuth_index[index] = position_data.azimuth_index;
    args.d_range_index[index] = position_data.range_index;

    const int range_int = static_cast<int>(position_data.range_index);
    const int azi_int = static_cast<int>(position_data.azimuth_index);

    // TODO could do reduction with shared memory instead of global atomics, but this likely won't be a bottleneck on
    // any GPU
    atomicMax(&result->max_azimuth, azi_int);
    atomicMin(&result->min_azimuth, azi_int);
    atomicMax(&result->max_range, range_int);
    atomicMin(&result->min_range, range_int);
}

Rectangle GetSourceRectangle(TcTileIndexPair tile_coordinates, GetSourceRectangleKernelArgs args,
                             PerThreadData* ctx) {
    dim3 block_dim{16, 16};
    dim3 grid_dim{static_cast<uint>(cuda::GetGridDim(block_dim.x, tile_coordinates.target_width)),
                  static_cast<uint>(cuda::GetGridDim(block_dim.y, tile_coordinates.target_height))};

    auto* h_result = ctx->h_source_rectangle_result.Get();
    h_result->min_range = h_result->min_azimuth = INT32_MAX;
    h_result->max_range = h_result->max_azimuth = INT32_MIN;

    auto* d_result = ctx->device_memory_arena.Alloc<SourceRectangeResult>();

    cuda::CopyAsyncH2D(d_result, h_result, ctx->stream);
    GetSourceRectangleKernel<<<grid_dim, block_dim, 0, ctx->stream>>>(tile_coordinates, args, d_result);
    cuda::CopyAsyncD2H(h_result, d_result, ctx->stream);
    // need to wait for async memcpy to complete
    CHECK_CUDA_ERR(cudaStreamSynchronize(ctx->stream));
    CHECK_CUDA_ERR(cudaGetLastError());

    if (h_result->max_azimuth == INT32_MIN) {
        return {};
    }

    const auto margin = BILINEAR_INTERPOLATION_MARGIN;

    const auto x_min = std::max(0, h_result->min_range - margin);
    const auto x_max = std::min(static_cast<int>(args.source_image_width), h_result->max_range + 2 * margin + 1);

    const auto y_min = std::max(0, h_result->min_azimuth - margin);
    const auto y_max = std::min(static_cast<int>(args.source_image_height), h_result->max_azimuth + 2 * margin + 1);

    return {x_min, y_min, x_max - x_min, y_max - y_min};
}
}  // namespace alus::terraincorrection
