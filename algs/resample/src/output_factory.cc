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

#include "output_factory.h"

#include <cstddef>
#include <string>
#include <string_view>

namespace {

constexpr size_t PADDING_VALUE_BOUND{10};

void AddTileNumberPadded(std::string& value, size_t no) {
    if (no < PADDING_VALUE_BOUND) {
        value.append("0");
    }
    value.append(std::to_string(no));
}

void AddTileNumbers(std::string& value, size_t x_no, size_t y_no) {
    value.append("_");
    AddTileNumberPadded(value, x_no);
    value.append("_");
    AddTileNumberPadded(value, y_no);
}
}  // namespace

namespace alus::resample {

std::string CreateResampledTilePath(std::string_view path_stem, size_t tile_x_no, size_t tile_y_no,
                                    std::string_view extension) {
    std::string path{path_stem};
    AddTileNumbers(path, tile_x_no, tile_y_no);
    if (!extension.empty()) {
        path.append(".").append(extension);
    }
    return path;
}

}  // namespace alus::resample
