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

#include <string_view>
#include <vector>

#include "tile.h"

namespace alus {

class IDataTileReadWriteBase {
protected:
    std::string_view file_name_;
    std::vector<int> band_map_;
    int band_count_{};
    // following fields get values from data
    int band_x_size_{};
    int band_y_size_{};
    int band_x_min_{};
    int band_y_min_{};
    std::string_view data_projection_{};
    std::vector<double> affine_geo_transform_;

public:
    IDataTileReadWriteBase(const std::string_view& file_name, const std::vector<int>& band_map, int band_count)
        : file_name_(file_name), band_map_(band_map), band_count_(band_count) {}
    IDataTileReadWriteBase(const std::string_view& file_name, const std::vector<int>& band_map, int band_count,
                           int band_x_size, int band_y_size, int band_x_min, int band_y_min,
                           const std::string_view& data_projection, const std::vector<double>& affine_geo_transform)
        : file_name_{file_name},
          band_map_{band_map},
          band_count_{band_count},
          band_x_size_{band_x_size},
          band_y_size_{band_y_size},
          band_x_min_{band_x_min},
          band_y_min_{band_y_min},
          data_projection_{data_projection},
          affine_geo_transform_{affine_geo_transform} {};
};

}  // namespace alus
