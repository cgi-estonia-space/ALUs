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

#include "resampling.h"

#include "shapes.h"

namespace alus {
namespace snapengine {
namespace resampling {

inline __device__ __host__ void AssignTileValuesImpl(Tile* tile, int new_width, int new_height, bool new_target,
                                                     bool new_scaled, float* data_buffer) {
    tile->width = new_width;
    tile->height = new_height;
    tile->scaled = new_scaled;
    tile->target = new_target;
    tile->data_buffer = data_buffer;
}

inline __device__ __host__ void SetRangeAzimuthIndicesImpl(ResamplingRaster& raster, double new_range_index,
                                                           double new_azimuth_index) {
    raster.range_index = new_range_index;
    raster.azimuth_index = new_azimuth_index;
}
}  // namespace resampling
}  // namespace snapengine
}  // namespace alus
