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

namespace alus::s1tbx {
class Point {
private:
    double x_, y_, z_;

public:
    Point();
    Point(double x, double y, double z);
    [[nodiscard]] double GetX() const { return x_; }
    [[nodiscard]] double GetY() const { return y_; }
    [[nodiscard]] double GetZ() const { return z_; }
    [[nodiscard]] Point Min(const Point& p) const;
    [[nodiscard]] double In(const Point& p) const;
    [[nodiscard]] double Norm() const;
    void SetX(double x);
    void SetY(double y);
    void SetZ(double z);
};
}  // namespace alus::s1tbx
