/**
 * This file is a filtered duplicate of a SNAP's
 * org.esa.snap.core.util.math.FXYSum.java
 * ported for native code.
 * Copied from (https://github.com/senbox-org/snap-engine). It was originally stated:
 *
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
#include "f_x_y_sum_bi_quadric.h"

namespace alus::snapengine {

BiQuadric::BiQuadric() : FXYSum(fxy_bi_quadratic_, 2 + 2) {}

BiQuadric::BiQuadric(const std::vector<double>& coefficients) : FXYSum(fxy_bi_quadratic_, 2 + 2, coefficients) {}

double BiQuadric::ComputeZ(double x, double y) {
    std::vector<double> c = GetCoefficients();
    return c.at(0) +
           (c.at(1) + (c.at(3) + (c.at(6) + c.at(8) * y) * y) * x + (c.at(4) + c.at(7) * y) * y) * x +  // NOLINT
           (c.at(2) + c.at(5) * y) * y;                                                                 // NOLINT
}

}  // namespace alus::snapengine