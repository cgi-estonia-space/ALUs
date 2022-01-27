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
#include <cmath>
#include <vector>

#include "shapes.h"

namespace alus::shapeutils {

inline Rectangle GetIntersection(const Rectangle& rectangle_1, const Rectangle& rectangle_2) {
    int target_x = rectangle_1.x;
    int target_y = rectangle_1.y;
    int64_t target_width = target_x + rectangle_1.width;
    int64_t target_height = target_y + rectangle_1.height;
    int64_t rx2 = rectangle_2.x + rectangle_2.width;
    int64_t ry2 = rectangle_2.y + rectangle_2.height;
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

inline std::vector<Rectangle> GenerateRectanglesForRaster(int raster_size_x, int raster_size_y, int rectangle_size_x,
                                                          int rectangle_size_y) {
    int x_rectangles =
        static_cast<int16_t>(std::ceil(static_cast<float>(raster_size_x) / static_cast<float>(rectangle_size_x)));
    int y_rectangles =
        static_cast<int16_t>(std::ceil(static_cast<float>(raster_size_y) / static_cast<float>(rectangle_size_y)));
    std::vector<Rectangle> rectangles;
    rectangles.reserve(static_cast<std::size_t>(y_rectangles) * static_cast<size_t>(x_rectangles));

    int y_max;
    int x_max;
    int x_min;
    int y_min;

    for (auto rectangle_y = 0; rectangle_y < y_rectangles; rectangle_y++) {
        if (rectangle_size_y >= raster_size_y) {
            y_min = 0;
            y_max = raster_size_y - 1;
        } else {
            if (rectangle_y == 0) {
                y_min = 0;
            } else {
                y_min = rectangle_size_y * rectangle_y;
            }

            if (rectangle_y == y_rectangles - 1) {
                y_max = raster_size_y - 1;
            } else {
                y_max = (y_min + rectangle_size_y) - 1;
            }
        }
        for (auto rectangle_x = 0; rectangle_x < x_rectangles; rectangle_x++) {
            if (rectangle_size_x >= raster_size_x) {
                x_min = 0;
                x_max = raster_size_x - 1;
            } else {
                if (rectangle_x == 0) {
                    x_min = 0;
                } else {
                    x_min = rectangle_size_x * rectangle_x;
                }

                if (rectangle_x == x_rectangles - 1) {
                    x_max = raster_size_x - 1;
                } else {
                    x_max = (x_min + rectangle_size_x) - 1;
                }
            }

            rectangles.push_back({x_min, y_min, x_max - x_min + 1, y_max - y_min + 1});
        }
    }

    return rectangles;
}

}  // namespace alus::shapeutils