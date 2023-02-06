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
    double tile_pixel_count_inverted_x;  // NUM_PIXELS_PER_TILE_INVERTED
    double tile_pixel_count_inverted_y;  // NUM_PIXELS_PER_TILE_INVERTED
    size_t tile_pixel_count_x;           // NUM_PIXELS_PER_TILE
    size_t tile_pixel_count_y;           // NUM_PIXELS_PER_TILE
    size_t grid_tile_count_x;            // NUM_X_TILES
    size_t grid_tile_count_y;            // NUM_Y_TILES
    size_t grid_total_width_pixels;      // RASTER_WIDTH
    size_t grid_total_height_pixels;     // RASTER_HEIGHT
    double no_data_value;
    double tile_pixel_size_deg_x;           // DEGREE_RES_BY_NUM_PIXELS_PER_TILE
    double tile_pixel_size_deg_y;           // DEGREE_RES_BY_NUM_PIXELS_PER_TILE
    double tile_pixel_size_deg_inverted_x;  // DEGREE_RES_BY_NUM_PIXELS_PER_TILE_INVERTED
    double tile_pixel_size_deg_inverted_y;  // DEGREE_RES_BY_NUM_PIXELS_PER_TILE_INVERTED
    double grid_max_lat;
    double grid_max_lon;
    double tile_lat_origin;
    double tile_lat_extent;
    double tile_lon_origin;
    double tile_lon_extent;
};

static_assert(sizeof(Property) % 8 == 0);

}  // namespace alus::dem