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
#include "topsar_deburst_rectangle_generator.h"

#include <cmath>

#include "custom/rectangle.h"

namespace alus {

TOPSARDeburstRectanglesGenerator::TOPSARDeburstRectanglesGenerator(int band_x_size, int band_y_size,
                                                                   int rectangle_x_size, int rectangle_y_size)
    : band_x_size_{band_x_size},
      band_y_size_{band_y_size},
      rectangle_x_size_{rectangle_x_size},
      rectangle_y_size_{rectangle_y_size} {}

// pre-generate tile attributes
std::vector<snapengine::custom::Rectangle> TOPSARDeburstRectanglesGenerator::GenerateRectangles() const {
    int x_rectangles = GetNumberOfRectanglesDim(band_x_size_, rectangle_x_size_);
    int y_rectangles = GetNumberOfRectanglesDim(band_y_size_, rectangle_y_size_);
    std::vector<snapengine::custom::Rectangle> rectangles;
    rectangles.reserve(static_cast<std::size_t>(y_rectangles * x_rectangles));

    int y_max;
    int x_max;
    int x_min;
    int y_min;

    for (auto rectangle_y = 0; rectangle_y < y_rectangles; rectangle_y++) {
        if (rectangle_y_size_ >= band_y_size_) {
            y_min = 0;
            y_max = band_y_size_ - 1;
        } else {
            if (rectangle_y == 0) {
                y_min = 0;
            } else {
                y_min = rectangle_y_size_ * rectangle_y;
            }

            if (rectangle_y == y_rectangles - 1) {
                y_max = band_y_size_ - 1;
            } else {
                y_max = (y_min + rectangle_y_size_) - 1;
            }
        }
        for (auto rectangle_x = 0; rectangle_x < x_rectangles; rectangle_x++) {
            if (rectangle_x_size_ >= band_x_size_) {
                x_min = 0;
                x_max = band_x_size_ - 1;
            } else {
                if (rectangle_x == 0) {
                    x_min = 0;
                } else {
                    x_min = rectangle_x_size_ * rectangle_x;
                }

                if (rectangle_x == x_rectangles - 1) {
                    x_max = band_x_size_ - 1;
                } else {
                    x_max = (x_min + rectangle_x_size_) - 1;
                }
            }

            rectangles.emplace_back(snapengine::custom::Rectangle(x_min, y_min, x_max - x_min + 1, y_max - y_min + 1));
        }
    }

    return rectangles;
}

int TOPSARDeburstRectanglesGenerator::GetNumberOfRectanglesDim(int band_size_dim, int rectangle_size_dim) {
    return static_cast<short>(std::ceil(static_cast<float>(band_size_dim) / static_cast<float>(rectangle_size_dim)));
}

std::vector<snapengine::custom::Rectangle> TOPSARDeburstRectanglesGenerator::GetRectangles() const {
    //    todo: think later maybe use member and provide reference?
    return GenerateRectangles();
}

}  // namespace alus