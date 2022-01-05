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
class Tile final {
    int x_max_{}, y_max_{}, x_min_{}, y_min_{};

public:
    Tile() = default;
    Tile(int x_max, int y_max, int x_min, int y_min) : x_max_(x_max), y_max_(y_max), x_min_(x_min), y_min_(y_min) {}
    [[nodiscard]] int GetXMin() const { return x_min_; }
    [[nodiscard]] int GetYMin() const { return y_min_; }
    [[nodiscard]] int GetXMax() const { return x_max_; }
    [[nodiscard]] int GetYMax() const { return y_max_; }
    [[nodiscard]] int GetXSize() const { return x_max_ - x_min_ + 1; }
    [[nodiscard]] int GetYSize() const { return y_max_ - y_min_ + 1; }
};
}  // namespace alus