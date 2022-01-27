/**
 * This file is a filtered duplicate of a SNAP's
 * org.esa.snap.engine_utilities.util.Maths.java
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
#pragma once

#include <vector>

#include "eigen3/Eigen/Dense"

namespace alus::snapengine {

class Maths {
public:
    /**
     * Get Vandermonde matrix constructed from a given array.
     *
     * @param d                   The given range distance array.
     * @param warp_polynomial_order The warp polynomial order.
     * @return The Vandermonde matrix.
     */
    static Eigen::MatrixXd CreateVandermondeMatrix(const std::vector<double>& d, int warp_polynomial_order);

    static std::vector<double> PolyFit(const Eigen::MatrixXd& A, std::vector<double> y);

    static double PolyVal(double t, const std::vector<double>& coeff);
};

}  // namespace alus::snapengine