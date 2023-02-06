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

#include "computation_metadata.h"
#include "cuda_copies.h"
#include "cuda_mem_arena.h"
#include "cuda_ptr.h"
#include "dem_property.h"
#include "dem_type.h"
#include "get_position.h"
#include "kernel_array.h"
#include "pointer_holders.h"
#include "raster_properties.h"
#include "resampling.h"

namespace alus::terraincorrection {

struct GetSourceRectangleKernelArgs {
    GetPositionMetadata get_position_metadata;
    bool use_avg_scene_height;
    double avg_scene_height;
    double dem_no_data_value;
    unsigned int source_image_width;
    unsigned int source_image_height;
    int diff_lat;
    GeoTransformParameters target_geo_transform;
    PointerArray srtm_3_tiles;
    const dem::Property* dem_property;
    dem::Type dem_type;
    double* d_azimuth_index;
    double* d_range_index;
};

struct TerrainCorrectionKernelArgs {
    unsigned int source_image_width;
    unsigned int source_image_height;
    float target_no_data_value;
    double* d_azimuth_index;
    double* d_range_index;
    snapengine::resampling::ResamplingRaster resampling_raster;
    cuda::KernelArray<float> d_target;
};

struct SourceRectangeResult {
    int min_azimuth;
    int max_azimuth;
    int min_range;
    int max_range;
};

struct PerThreadData {
    bool init_done = false;
    bool use_pinned_memory = false;
    PagedOrPinnedHostPtr<float> h_target_tile;
    PagedOrPinnedHostPtr<float> h_source_tile;
    size_t source_buffer_size = 0;
    PagedOrPinnedHostPtr<SourceRectangeResult> h_source_rectangle_result;
    PagedOrPinnedHostPtr<snapengine::resampling::Tile> h_resampling_tile;

    alus::cuda::MemArena device_memory_arena;
    cuda::CudaPtr<float> d_source_buffer;
    cudaStream_t stream;
};

void CalculateVelocitiesAndPositions(int source_image_height, double first_line_utc, double line_time_interval,
                                     cuda::KernelArray<snapengine::OrbitStateVectorComputation> vectors,
                                     cuda::KernelArray<snapengine::PosVector> velocities,
                                     cuda::KernelArray<snapengine::PosVector> positions);

cudaError_t LaunchTerrainCorrectionKernel(TcTileCoordinates tc_tile_coordinates, TerrainCorrectionKernelArgs args,
                                          float* h_target_buffer, cudaStream_t stream);

/**
 * Custom method for calculating source rectangle of a given target tile.
 *
 * @param tile_coordinates Struct containing pixel indices of the target tile.
 * @param args Terrain Correction arguments.
 * @return Rectangle containing pixels corresponding to the given target tile.
 */
Rectangle GetSourceRectangle(TcTileCoordinates tile_coordinates, GetSourceRectangleKernelArgs args, PerThreadData* ctx);
}  // namespace alus::terraincorrection
