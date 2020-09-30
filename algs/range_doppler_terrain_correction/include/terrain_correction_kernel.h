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
#include "get_position.h"
#include "kernel_array.h"
#include "pointer_holders.h"
#include "raster_properties.hpp"
#include "resampling.h"

namespace alus {
namespace terraincorrection {
struct TerrainCorrectionKernelArgs {
    unsigned int source_image_width;
    unsigned int source_image_height;
    double dem_no_data_value;
    double target_no_data_value;
    double avg_scene_height;
    GeoTransformParameters target_geo_transform;
    bool use_avg_scene_height;
    int diff_lat;
    ComputationMetadata metadata;
    GetPositionMetadata get_position_metadata;
    snapengine::resampling::ResamplingRaster resampling_raster;
    snapengine::resampling::ResamplingIndex resampling_index;
    const PointerHolder* srtm_3_tiles;
};

void CalculateVelocitiesAndPositions(int source_image_height, double first_line_utc,
                                     double line_time_interval,
                                     cuda::KernelArray<snapengine::OrbitStateVectorComputation> vectors,
                                     cuda::KernelArray<snapengine::PosVector> velocities,
                                     cuda::KernelArray<snapengine::PosVector> positions);

cudaError_t LaunchTerrainCorrectionKernel(TcTile tile, TerrainCorrectionKernelArgs args);

bool GetSourceRectangle(TcTile& tile, GeoTransformParameters target_geo_transform, double dem_no_data_value,
                        size_t source_image_width, size_t source_image_height, GetPositionMetadata get_position_metadata,
                        Rectangle& source_rectangle, const PointerHolder* srtm_3_tiles, double avg_scene_height,
                        bool use_avg_height);
}  // namespace terraincorrection
}  // namespace alus