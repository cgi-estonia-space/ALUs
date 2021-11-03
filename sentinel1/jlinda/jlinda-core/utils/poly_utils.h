/**
 * This file is a filtered duplicate of a SNAP's
 * org.jlinda.core.utils.PolyUtils.java
 * ported for native code.
 * Copied from (https://github.com/senbox-org/s1tbx). It was originally stated:
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

#include <eigen3/Eigen/Dense>

namespace alus {

class PolyUtils {
public:
    [[nodiscard]] static double PolyVal1D(double x, const std::vector<double>& coeffs);
    [[nodiscard]] static std::vector<double> Solve33(const std::vector<std::vector<double>>& a, const std::vector<double>& rhs);

    static Eigen::VectorXd PolyFit(const Eigen::VectorXd& xvals, const Eigen::VectorXd& yvals, int order);

    static Eigen::VectorXd Normalize(const Eigen::VectorXd& t);

    /**
     * polyfit
     * <p/>
     * Compute coefficients of x=a0+a1*t+a2*t^2+a3*t3 polynomial
     * for orbit interpolation.  Do this to facilitate a method
     * in case only a few datapoints are given.
     * Data t is normalized approximately [-x,x], then polynomial
     * coefficients are computed.  For poly_val this is repeated
     * see getxyz, etc.
     * <p/>
     * input:
     * - matrix by getdata with time and position info
     * output:
     * - matrix with coeff.
     * (input for interp. routines)
     */
    static std::vector<double> PolyFitNormalized(std::vector<double> t, std::vector<double> y, int degree);

    static Eigen::VectorXd PolyFitNormalized(const Eigen::VectorXd& t, const Eigen::VectorXd& y, int degree);
};

}  // namespace alus
