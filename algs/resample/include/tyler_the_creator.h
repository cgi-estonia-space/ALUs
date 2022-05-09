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

#include <vector>

#include "raster_properties.h"

namespace alus::resample {

struct TileConstruct {
    alus::RasterDimension image_dimension;
    alus::GeoTransformParameters image_gt;
    alus::RasterDimension tile_dimension;
    size_t overlap;
};

struct TileProperties {
    alus::PixelPosition offset;
    alus::GeoTransformParameters gt;
    alus::RasterDimension dimension;
    size_t tile_no_x;
    size_t tile_no_y;
};

std::vector<TileProperties> CreateTiles(const TileConstruct& construct_parameters);

}  // namespace alus::resample