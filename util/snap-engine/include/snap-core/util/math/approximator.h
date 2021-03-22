/**
 * This file is a filtered duplicate of a SNAP's
 * org.esa.snap.core.util.math.Approximator.java
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

#include <functional>
#include <memory>
#include <vector>

#include <eigen3/Eigen/Dense>

#include "f_x_y_sum.h"
#include "snap-core/util/math/i_f_x_y.h"

namespace alus {
namespace snapengine {
/**
 * A utility class which can be used to find approximation functions for a given dataset.
 *
 * orginal java version author Norman Fomferra
 */
class Approximator {
private:
    /**
     * Solves the matrix equation A x = b by means of singular value decomposition.
     *
     * @param a the matrix A
     * @param b the vector b
     * @param x the solution vector x.
     */
    static void Solve2(const Eigen::MatrixXd& a, const Eigen::VectorXd& b, Eigen::VectorXd& x);
    //    static void Solve2(std::vector<std::vector<double>> a, std::vector<double> b, std::vector<double> x);
public:
    /**
     * Solves a linear equation system with each term having the form c * f(x,y). The method finds the coefficients
     * <code>c[0]</code> to <code>c[n-1]</code> with <code>n = f.length</code> for an approximation function z'(x,y) =
     * <code>c[0]*f[0](x,y) + c[1]*f[1](x,y) + c[2]*f[2](x,y) + ... + c[n-1]*f[n-1](x,y)</code> which approximates the
     * given data vector x<sub>i</sub>=<code>data[i][0]</code>, y<sub>i</sub>=<code>data[i][1]</code>,
     * z<sub>i</sub>=<code>data[i][2]</code> with i = <code>0</code> to <code>data.length-1</code>.
     *
     * @param data    an array of values of the form <code>{{x1,y1,z1}, {x2,y2,z2}, {x3,y3,z3}, ...} </code>
     * @param indices the co-ordinate indices vector, determining the indices of x,y,z within <code>data</code>. If
     *                <code>null</code> then <code>indices</code> defaults to <code>{0, 1, 2}</code>.
     * @param f       the function vector, each function has the form z=f(x,y)
     * @param c       the resulting coefficient vector, must have the same size as the function vector
     */
    static void ApproximateFXY(const std::vector<std::vector<double>>& data, std::vector<int> indices,
                               const std::vector<std::reference_wrapper<IFXY>>& f, std::vector<double>& c);

    /**
     * Returns the root mean square error (RMSE) for the approximation of the given data with a function given by
     * z(x,y) = <code>c[0]*f[0](x,y) + c[1]*f[1](x,y) + c[2]*f[2](x,y) + ... + c[n-1]*f[n-1](x,y)</code>.
     *
     * @param data    an array of values of the form <code>{{x1,y1,z1}, {x2,y2,z1}, {x3,y3,z1}, ...} </code>
     * @param indices the co-ordinate indices vector, determining the indices of x,y,z within <code>data</code>. If
     *                <code>null</code> then <code>indices</code> defaults to <code>{0, 1, 2}</code>.
     * @param f       the function vector, each function has the form z=f(x,y)
     * @param c       the coefficient vector, must have the same size as the function vector
     */
    static std::vector<double> ComputeErrorStatistics(const std::vector<std::vector<double>>& data,
                                                      std::vector<int> indices,
                                                      const std::vector<std::reference_wrapper<IFXY>>& f,
                                                      const std::vector<double>& c);
};

}  // namespace snapengine
}  // namespace alus
