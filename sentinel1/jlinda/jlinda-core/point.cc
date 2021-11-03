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
#include "point.h"

#include <cmath>

namespace alus::s1tbx {

Point::Point() {
    this->x_ = 0;
    this->y_ = 0;
    this->z_ = 0;
}

Point::Point(double x, double y, double z) : x_{x}, y_{y}, z_{z} {}

void Point::SetX(double x) { this->x_ = x; }
void Point::SetY(double y) { this->y_ = y; }
void Point::SetZ(double z) { this->z_ = z; }

Point Point::Min(const Point &p) const {
    double dx = this->x_ - p.x_;
    double dy = this->y_ - p.y_;
    double dz = this->z_ - p.z_;
    return Point(dx, dy, dz);
}

// inner product
double Point::In(const Point &p) const {
    double dx = x_ * p.x_;
    double dy = y_ * p.y_;
    double dz = z_ * p.z_;
    return (dx + dy + dz);
}

double Point::Norm() const { return sqrt(x_ * x_ + y_ * y_ + z_ * z_); }

}  // namespace alus
