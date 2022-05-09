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

#include <array>
#include <string_view>

namespace alus::resample::sentinel2 {

constexpr std::array<std::string_view, 13> BANDS{"B1", "B2",  "B3", "B4",  "B5",  "B6", "B7",
                                                 "B8", "B8A", "B9", "B10", "B11", "B12"};
constexpr size_t SUBDATASET_COUNT{4};
constexpr std::string_view PRODUCT_URI_KEY{"PRODUCT_URI"};
constexpr std::string_view SUBDATASET_NAME_IDENTIFIER{"_NAME=SENTINEL2_"};
constexpr std::string_view BANDNAME_METADATA_KEY{"BANDNAME"};
constexpr std::string_view TRUE_COLOR_IMAGE_IDENTIFIER{".xml:TCI:"};
constexpr size_t PIXEL_BYTE_SIZE{2};
}  // namespace alus::resample::sentinel2