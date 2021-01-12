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

#include <climits>

#include "shapes.h"

namespace alus {
namespace shapeutils {

inline Rectangle GetIntersection(const Rectangle& rectangle_1, const Rectangle& rectangle_2) {
    int target_x = rectangle_1.x;
    int target_y = rectangle_1.y;
    long target_width = target_x + rectangle_1.width;
    long target_height = target_y + rectangle_1.height;
    long rx2 = rectangle_2.x + rectangle_2.width;
    long ry2 = rectangle_2.y + rectangle_2.height;
    if (target_x < rectangle_2.x) {
        target_x = rectangle_2.x;
    }
    if (target_y < rectangle_2.y) {
        target_y = rectangle_2.y;
    }
    if (target_width > rx2) {
        target_width = rx2;
    }
    if (target_height > ry2) {
        target_height = ry2;
    }
    target_width -= target_x;
    target_height -= target_y;
    if (target_width < INT_MIN) {
        target_width = INT_MIN;
    }
    if (target_height < INT_MIN) {
        target_height = INT_MIN;
    }
    return {target_x, target_y, static_cast<int>(target_width), static_cast<int>(target_height)};
}
}  // namespace shapeutils
}  // namespace alus