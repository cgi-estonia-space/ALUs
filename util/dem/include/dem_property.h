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

namespace alus::dem {

struct Property {
    double pixels_per_tile_inverted_x_axis;  // NUM_PIXELS_PER_TILE_INVERTED
    double pixels_per_tile_inverted_y_axis;  // NUM_PIXELS_PER_TILE_INVERTED
    size_t pixels_per_tile_x_axis;           // NUM_PIXELS_PER_TILE
    size_t pixels_per_tile_y_axis;           // NUM_PIXELS_PER_TILE
    size_t tiles_x_axis;                     // NUM_X_TILES
    size_t tiles_y_axis;                     // NUM_Y_TILES
    size_t raster_width;                     // RASTER_WIDTH
    size_t raster_height;                    // RASTER_HEIGHT
    double no_data_value;
    double pixel_size_degrees_x_axis;           // DEGREE_RES_BY_NUM_PIXELS_PER_TILE
    double pixel_size_degrees_y_axis;           // DEGREE_RES_BY_NUM_PIXELS_PER_TILE
    double pixel_size_degrees_inverted_x_axis;  // DEGREE_RES_BY_NUM_PIXELS_PER_TILE_INVERTED
    double pixel_size_degrees_inverted_y_axis;  // DEGREE_RES_BY_NUM_PIXELS_PER_TILE_INVERTED
    double lat_coverage;
    double lon_coverage;
    double lat_origin;
    double lat_extent;
    double lon_origin;
    double lon_extent;
};

static_assert(sizeof(Property) % 8 == 0);

}  // namespace alus::dem