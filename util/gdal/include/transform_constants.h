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

namespace alus::transform {

// These are the TOP LEFT / UPPER LEFT coordinates of the image.
static constexpr int TRANSFORM_LON_ORIGIN_INDEX{0};    // Or X origin
static constexpr int TRANSFORM_PIXEL_X_SIZE_INDEX{1};  // Or pixel width
static constexpr int TRANSFORM_ROTATION_1{2};
static constexpr int TRANSFORM_LAT_ORIGIN_INDEX{3};  // Or Y origin
static constexpr int TRANSFORM_ROTATION_2{4};
static constexpr int TRANSFORM_PIXEL_Y_SIZE_INDEX{5};  // Or pixel height
static constexpr size_t GEOTRANSFORM_ARRAY_LENGTH{6};
}  // namespace alus::transform