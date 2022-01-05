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
#pragma once

#include <functional>
#include <memory>
#include <vector>

#include "i_f_x_y.h"

namespace alus::snapengine {

/**
 * The class <code>FXYSum</code> represents a sum of function terms <i>sum(c[i] * f[i](x,y), i=0, n-1)</i>
 * where the vector <i>c</i> is a <code>double</code> array of constant coefficients and the vector <i>f</i>
 * is an array of functions of type <code>{@link FXY}</code> in <i>x</i> and <i>y</i>.
 * <p>
 * The vector <i>c</i> of constants is set by the {@link FXYSum#approximate(double[][], int[])} method.
 * <p>
 * After vector <i>c</i> is set, the actual function values <i>z(x,y)</i> are retrieved using the
 * {@link FXYSum#computeZ(double, double)} method.
 *
 * original java version author Norman Fomferra
 */
class FXYSum {
private:
    std::vector<std::reference_wrapper<IFXY>> f_;
    std::vector<double> c_;
    int order_;
    std::vector<double> error_statistics_;

public:
    static std::vector<std::reference_wrapper<IFXY>> fxy_linear_;
    static std::vector<std::reference_wrapper<IFXY>> fxy_bi_linear_;
    static std::vector<std::reference_wrapper<IFXY>> fxy_quadratic_;
    static std::vector<std::reference_wrapper<IFXY>> fxy_bi_quadratic_;
    static std::vector<std::reference_wrapper<IFXY>> fxy_cubic_;
    static std::vector<std::reference_wrapper<IFXY>> fxy_bi_cubic_;
    static std::vector<std::reference_wrapper<IFXY>> fxy_4_th_;
    static std::vector<std::reference_wrapper<IFXY>> fxy_bi_4_th_;

    /**
     * Creates a {@link FXYSum} by the given order and coefficients.
     * <p><b>Note: </b>
     * This factory method supprots only the creation of instances of the following FXYSum classes:
     * <ul>
     * <li>{@link Linear} - order = 1 ; number of coefficients = 3</li>
     * <li>{@link BiLinear} - order = 2 ; number of coefficients = 4</li>
     * <li>{@link Quadric} - order = 2 ; number of coefficients = 6</li>
     * <li>{@link BiQuadric} - order = 4 ; number of coefficients = 9</li>
     * <li>{@link Cubic} - order = 3 ; number of coefficients = 10</li>
     * <li>{@link BiCubic} - order = 6 ; number of coefficients = 16</li>
     * </ul>
     *
     *
     * @param order        the order of the sum
     * @param coefficients the coefficients
     *
     * @return returns a FXYSum instance, or <code>null</code> if the resulting instance is one of the supported.
     */
    static std::shared_ptr<FXYSum> CreateFXYSum(int order, const std::vector<double>& coefficients);

    /**
     * Creates a copy of the given {@link FXYSum fxySum}.
     *
     * @param fxySum the {@link FXYSum} to copy
     *
     * @return a copy of the given {@link FXYSum}
     */
    static std::shared_ptr<FXYSum> CreateCopy(const std::shared_ptr<FXYSum>& fxy_sum);

    /**
     * Computes <i>z(x,y) = sum(c[i] * f[i](x,y), i = 0, n - 1)</i>.
     *
     * @param f the function vector
     * @param c the coeffcient vector
     * @param x the x value
     * @param y the y value
     *
     * @return the z value
     */
    static double ComputeZ(const std::vector<std::reference_wrapper<IFXY>>& f, const std::vector<double>& c, double x,
                           double y);

    /**
     * Constructs a new sum of terms <i>sum(c[i] * f[i](x,y), i=0, n-1)</i> for the given vector of functions.
     * The vector <i>c</i> is initally set to zero and will remeain zero until the method {@link
     * #approximate(double[][], int[])} is performed with given data on this function sum.
     *
     * @param functions the vector F of functions
     */
    explicit FXYSum(const std::vector<std::reference_wrapper<IFXY>>& functions);

    /**
     * Constructs a new sum of terms <i>sum(c[i] * f[i](x,y), i=0, n-1)</i> for the given vector of functions and a
     * polynomial order. The vector <i>c</i> is initally set to zero and will remeain zero until the method {@link
     * #approximate(double[][], int[])} is performed with given data on this function sum.
     *
     * @param functions the vector F of functions
     * @param order     the polynomial order (for descriptive purpose only), -1 if unknown
     */
    FXYSum(const std::vector<std::reference_wrapper<IFXY>>& functions, int order);

    /**
     * Constructs a new sum of terms <i>sum(c[i] * f[i](x,y), i=0, n-1)</i> for the given vector of functions and a
     * polynomial order. The vector <i>c</i> is set by the <code>coefficients</code> parameter. The coefficients will be
     * recalculated if the method {@link #approximate(double[][], int[])} is called.
     *
     * @param functions    the vector F of functions
     * @param order        the polynomial order (for descriptive purpose only), -1 if unknown
     * @param coefficients the vector <i>c</i>
     */
    FXYSum(const std::vector<std::reference_wrapper<IFXY>>& functions, int order,
           const std::vector<double>& coefficients);

    /**
     * Gets the number <i>n</i> of terms <i>c[i] * f[i](x,y)</i>.
     *
     * @return the number of function terms
     */
    int GetNumTerms() { return f_.size(); }

    /**
     * Gets the vector <i>f</i> of functions elements <i>f[i](x,y)</i>.
     *
     * @return the vector F of functions
     */
    std::vector<std::reference_wrapper<IFXY>> GetFunctions() { return f_; }

    /**
     * Gets the vector <i>c</i> of coefficient elements <i>c[i]</i>.
     *
     * @return the vector F of functions
     */
    std::vector<double> GetCoefficients() { return c_; }

    /**
     * Gets the polynomial order, if any.
     *
     * @return the polynomial order or -1 if unknown
     */
    [[nodiscard]] int GetOrder() const { return order_; }

    /**
     * Gets the root mean square error.
     *
     * @return the root mean square error
     */
    double GetRootMeanSquareError() { return error_statistics_.at(0); }

    /**
     * Gets the maximum, absolute error of the approximation.
     *
     * @return the maximum, absolute error
     */
    double GetMaxError() { return error_statistics_.at(1); }

    /**
     * Computes this sum of function terms <i>z(x,y) = sum(c[i] * f[i](x,y), i=0, n-1)</i>.
     * The method will return zero unless the {@link #approximate(double[][], int[])} is called in order to set
     * the coefficient vector <i>c</i>.
     *
     * @param x the x value
     * @param y the y value
     *
     * @return the z value
     *
     * @see #computeZ(FXY[], double[], double, double)
     */
    virtual double ComputeZ(double x, double y) { return ComputeZ(f_, c_, x, y); }

    /**
     * Approximates the given data points <i>x,y,z</i> by this sum of function terms so that <i>z ~ sum(c[i] *
     * f[i](x,y), i=0, n-1)</i>. The method also sets the error statistics which can then be retrieved by the {@link
     * #getRootMeanSquareError()} and {@link #getMaxError()} methods.
     *
     * @param data    an array of values of the form <i>{{x1,y1,z1}, {x2,y2,z2}, {x3,y3,z3}, ...} </i>
     * @param indices an array of coordinate indices, determining the indices of <i>x</i>, <i>y</i> and <i>z</i> within
     * a <code>data</code> element. If <code>null</code> then <code>indices</code> defaults to the array <code>{0, 1,
     * 2}</code>.
     *
     * @see Approximator#approximateFXY(double[][], int[], FXY[], double[])
     * @see Approximator#computeErrorStatistics(double[][], int[], FXY[], double[])
     * @see #computeZ(double, double)
     */
    void Approximate(const std::vector<std::vector<double>>& data, const std::vector<int>& indices);

    virtual ~FXYSum() = default;
};

}  // namespace alus::snapengine