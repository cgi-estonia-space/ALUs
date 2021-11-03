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
#include "maths.h"

#include <cmath>

namespace alus {
namespace snapengine {

Eigen::MatrixXd Maths::CreateVandermondeMatrix(std::vector<double> d, int warp_polynomial_order) {
    auto n = d.size();
    Eigen::MatrixXd array(n, warp_polynomial_order + 1);
    for (int i = 0; i < (int)n; i++) {
        for (int j = 0; j <= warp_polynomial_order; j++) {
            array(i, j) = pow(d.at(i), (double)j);
        }
    }
    return array;
}

std::vector<double> Maths::PolyFit(Eigen::MatrixXd A, std::vector<double> y) {
    //    todo:this might not be optimal solution, just needed it to work
    auto Q = A.householderQr();
    // std::vector to eigen vector
    double* ptr = &y[0];
    Eigen::Map<Eigen::VectorXd> yvals(ptr, y.size());
    Eigen::VectorXd result = Q.solve(yvals);
    // eigen to std::vector
    return std::vector<double>(result.data(), result.data() + result.rows() * result.cols());
}

double Maths::PolyVal(double t, std::vector<double> coeff) {
    double val = 0.0;
    int i = coeff.size() - 1;
    //        todo::looks like some logical issue because size can only be >=0
    while (i >= 0) {
        val = val * t + coeff.at(i--);
    }
    return val;
}

}  // namespace snapengine
}  // namespace alus