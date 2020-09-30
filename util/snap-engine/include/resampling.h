#pragma once

#include "cuda_util.cuh"
#include "shapes.h"

namespace alus {
namespace snapengine {
namespace resampling {

struct ResamplingIndex {
    double x;
    double y;
    int width;
    int height;
    double i0;
    double j0;
    double *i;
    double *j;
    double *ki;
    double *kj;
};

struct Tile {
    int x_0;
    int y_0;
    int width;
    int height;
    bool target;
    bool scaled;
    double *data_buffer;
};

struct ResamplingRaster {
    double range_index;
    double azimuth_index;
    int sub_swath_index = -1;
    int min_x;
    int min_y;
    alus::Rectangle *source_rectangle;
    alus::snapengine::resampling::Tile *source_tile_i;
};

struct TileData {
    alus::snapengine::resampling::ResamplingRaster *resampling_raster;
    alus::snapengine::resampling::Tile *source_tile;
    alus::snapengine::resampling::ResamplingIndex *image_resampling_index;
};
}  // namespace resampling
}  // namespace snapengine
}  // namespace alus