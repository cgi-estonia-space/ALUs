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

#include <memory>
#include <vector>

#include "custom/rectangle.h"

namespace alus {

class TOPSARDeburstRectanglesGenerator {
private:
    int band_x_size_;
    int band_y_size_;
    int rectangle_x_size_;
    int rectangle_y_size_;

    [[nodiscard]] static int GetNumberOfRectanglesDim(int band_size_dim, int rectangle_size_dim);
    [[nodiscard]] std::vector<snapengine::custom::Rectangle> GenerateRectangles() const;

public:
    TOPSARDeburstRectanglesGenerator(int band_x_size, int band_y_size, int rectangle_x_size, int rectangle_y_size);
    [[nodiscard]] std::vector<snapengine::custom::Rectangle> GetRectangles() const;
};
}  // namespace alus