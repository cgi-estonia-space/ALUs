#include <utility>

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

namespace alus {

struct BandParams {
    std::vector<int> band_map;
    int band_count;
    int band_x_size{0};
    int band_y_size{0};
    int band_x_min{0};
    int band_y_min{0};
    BandParams(std::vector<int>  band_map, int band_count)
        : band_map(std::move(band_map)), band_count(band_count){}
    BandParams(std::vector<int>  band_map, int band_count, int band_x_size, int band_y_size, int band_x_min,
               int band_y_min)
        : band_map(std::move(band_map)),
          band_count(band_count),
          band_x_size(band_x_size),
          band_y_size(band_y_size),
          band_x_min(band_x_min),
          band_y_min(band_y_min) {}
};

}  // namespace alus