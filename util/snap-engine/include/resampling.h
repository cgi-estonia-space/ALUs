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
    size_t width;
    size_t height;
    bool target;
    bool scaled;
    double *data_buffer;
};

struct ResamplingRaster {
    double range_index;
    double azimuth_index;
    int sub_swath_index = -1;
    Rectangle source_rectangle;
    Tile *source_tile_i;
    bool source_rectangle_calculated;
};
}  // namespace resampling
}  // namespace snapengine
}  // namespace alus