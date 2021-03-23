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

namespace alus {
namespace snapengine {
namespace custom {

struct Rectangle {
    Rectangle() = default;
    Rectangle(int x, int y, int width, int height) : x(x), y(y), width(width), height(height) {}
    explicit Rectangle(const std::shared_ptr<Rectangle>& rectangle) {
        x = rectangle->x;
        y = rectangle->y;
        width = rectangle->width;
        height = rectangle->height;
    }
    int x, y, width, height;
};

}  // namespace custom
}  // namespace snapengine
}  // namespace alus
