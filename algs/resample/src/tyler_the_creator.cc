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

#include "tyler_the_creator.h"

#include <algorithm>
#include <stdexcept>
#include <string>

#include "raster_properties.h"

namespace {

void CheckTileSize(alus::RasterDimension image_dim, alus::RasterDimension tile_dim, size_t overlap) {
    if (tile_dim.columnsX < 1 || tile_dim.rowsY < 1) {
        throw std::invalid_argument("Tile dimensions (" + std::to_string(tile_dim.columnsX) + "," +
                                    std::to_string(tile_dim.rowsY) + ") are not correct.");
    }

    if (tile_dim.columnsX + static_cast<int>(overlap) > image_dim.columnsX ||
        tile_dim.rowsY + static_cast<int>(overlap) > image_dim.rowsY) {
        throw std::invalid_argument("Tile dimensions (" + std::to_string(tile_dim.columnsX) + "," +
                                    std::to_string(tile_dim.rowsY) + ") including overlap (" + std::to_string(overlap) +
                                    ")" + " are exceeding image dimensions (" + std::to_string(image_dim.columnsX) +
                                    "," + std::to_string(image_dim.rowsY) + ").");
    }

    if ((tile_dim.columnsX / 2) <= static_cast<int>(overlap) || (tile_dim.rowsY / 2) <= static_cast<int>(overlap)) {
        throw std::invalid_argument("Tile dimensions (" + std::to_string(tile_dim.columnsX) + "," +
                                    std::to_string(tile_dim.rowsY) + ") cannot accommodate overlap (" +
                                    std::to_string(overlap) + ").");
    }
}

}  // namespace

namespace alus::resample {

std::vector<TileProperties> CreateTiles(const TileConstruct& construct_parameters) {
    CheckTileSize(construct_parameters.image_dimension, construct_parameters.tile_dimension,
                  construct_parameters.overlap);

    std::vector<TileProperties> tiles;
    const auto image_dim_x = static_cast<size_t>(construct_parameters.image_dimension.columnsX);
    const auto image_dim_y = static_cast<size_t>(construct_parameters.image_dimension.rowsY);
    const auto pixel_size_lat = construct_parameters.image_gt.pixelSizeLat;
    const auto pixel_size_lon = construct_parameters.image_gt.pixelSizeLon;
    const auto step_x = construct_parameters.tile_dimension.columnsX - construct_parameters.overlap;
    const auto step_y = construct_parameters.tile_dimension.rowsY - construct_parameters.overlap;

    auto lat = construct_parameters.image_gt.originLat;
    size_t tile_y{};
    size_t tile_no_y{1};
    while (tile_y < image_dim_y) {
        auto tile_y_end = tile_y + construct_parameters.tile_dimension.rowsY;
        tile_y_end = std::clamp(tile_y_end, tile_y, image_dim_y);

        auto lon = construct_parameters.image_gt.originLon;
        size_t tile_x{};
        size_t tile_no_x{1};
        while (tile_x < image_dim_x) {
            auto tile_x_end = tile_x + construct_parameters.tile_dimension.columnsX;
            tile_x_end = std::clamp(tile_x_end, tile_x, image_dim_x);

            TileProperties prop{{static_cast<int>(tile_x), static_cast<int>(tile_y)},
                                {lon, lat, pixel_size_lon, pixel_size_lat},
                                {static_cast<int>(tile_x_end - tile_x), static_cast<int>(tile_y_end - tile_y)},
                                tile_no_x,
                                tile_no_y};
            tiles.push_back(prop);

            lon += step_x * pixel_size_lon;
            tile_x += step_x;
            tile_no_x++;
        }

        lat += step_y * pixel_size_lat;
        tile_y += step_y;
        tile_no_y++;
    }

    return tiles;
}

}  // namespace alus::resample
