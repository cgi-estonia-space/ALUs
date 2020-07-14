#pragma once

#include <cmath>

#include "resampling.h"

#include "shapes.h"

namespace alus {
namespace snapengine {
namespace resampling {

inline __device__ __host__ void AssignTileValuesImpl(
    Tile &tile, int new_width, int new_height, bool new_target, bool new_scaled, double *data_buffer) {
    tile.width = new_width;
    tile.height = new_height;
    tile.scaled = new_scaled;
    tile.target = new_target;
    tile.data_buffer = data_buffer;
}

inline __device__ __host__ void SetRangeAzimuthIndicesImpl(ResamplingRaster &raster,
                                                           double new_range_index,
                                                           double new_azimuth_index) {
    raster.range_index = new_range_index;
    raster.azimuth_index = new_azimuth_index;
}

inline __device__ __host__ void SetSourceRectangleImpl(ResamplingRaster &raster, alus::Rectangle &rectangle) {
    raster.source_rectangle = &rectangle;
    raster.min_x = rectangle.x;
    raster.min_y = rectangle.y;
}
}  // namespace resampling
}  // namespace snapengine
}  // namespace alus
