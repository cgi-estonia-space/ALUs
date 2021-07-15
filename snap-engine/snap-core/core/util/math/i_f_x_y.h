/**
 * This file is a filtered duplicate of a SNAP's
 * org.esa.snap.core.util.math.FXY.java
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
#include <string>

namespace alus {
namespace snapengine {

/**
 * Represents a function <i>f(x,y)</i>.
 *
 * original java version author Norman Fomferra
 */
class IFXY {
public:
    /**
     * The function <i>f(x,y) = x <sup>4</sup> y <sup>4</sup></i>
     */
    static const std::reference_wrapper<IFXY> X4Y4;

    /**
     * The function <i>f(x,y) = x <sup>4</sup> y <sup>3</sup></i>
     */
    static const std::reference_wrapper<IFXY> X4Y3;
    /**
     * The function <i>f(x,y) = x <sup>3</sup> y <sup>4</sup></i>
     */
    static const std::reference_wrapper<IFXY> X3Y4;
    /**
     * The function <i>f(x,y) = x <sup>4</sup> y <sup>2</sup></i>
     */
    static const std::reference_wrapper<IFXY> X4Y2;
    /**
     * The function <i>f(x,y) = x <sup>2</sup> y <sup>4</sup></i>
     */
    static const std::reference_wrapper<IFXY> X2Y4;
    /**
     * The function <i>f(x,y) = x <sup>4</sup> y </i>
     */
    static const std::reference_wrapper<IFXY> X4Y;
    /**
     * The function <i>f(x,y) = x y <sup>4</sup></i>
     */
    static const std::reference_wrapper<IFXY> XY4;
    /**
     * The function <i>f(x,y) = x <sup>4</sup> </i>
     */
    static const std::reference_wrapper<IFXY> X4;
    /**
     * The function <i>f(x,y) = y <sup>4</sup></i>
     */
    static const std::reference_wrapper<IFXY> Y4;
    /**
     * The function <i>f(x,y) = x <sup>3</sup> y <sup>3</sup></i>
     */
    static const std::reference_wrapper<IFXY> X3Y3;
    /**
     * The function <i>f(x,y) = x <sup>3</sup> y <sup>2</sup></i>
     */
    static const std::reference_wrapper<IFXY> X3Y2;
    /**
     * The function <i>f(x,y) = x <sup>2</sup> y <sup>3</sup></i>
     */
    static const std::reference_wrapper<IFXY> X2Y3;
    /**
     * The function <i>f(x,y) = x <sup>3</sup> y</i>
     */
    static const std::reference_wrapper<IFXY> X3Y;
    /**
     * The function <i>f(x,y) = x <sup>2</sup> y <sup>2</sup></i>
     */
    static const std::reference_wrapper<IFXY> X2Y2;
    /**
     * The function <i>f(x,y) = x y <sup>3</sup></i>
     */
    static const std::reference_wrapper<IFXY> XY3;
    /**
     * The function <i>f(x,y) = x <sup>3</sup></i>
     */
    static const std::reference_wrapper<IFXY> X3;
    /**
     * The function <i>f(x,y) = x <sup>2</sup> y</i>
     */
    static const std::reference_wrapper<IFXY> X2Y;
    /**
     * The function <i>f(x,y) = x y <sup>2</sup></i>
     */
    static const std::reference_wrapper<IFXY> XY2;
    /**
     * The function <i>f(x,y) = y <sup>3</sup></i>
     */
    static const std::reference_wrapper<IFXY> Y3;
    /**
     * The function <i>f(x,y) = x <sup>2</sup></i>
     */
    static const std::reference_wrapper<IFXY> X2;
    /**
     * The function <i>f(x,y) = x y</i>
     */
    static const std::reference_wrapper<IFXY> XY;
    /**
     * The function <i>f(x,y) = y <sup>2</sup></i>
     */
    static const std::reference_wrapper<IFXY> Y2;
    /**
     * The function <i>f(x,y) = x</i>
     */
    static const std::reference_wrapper<IFXY> X;
    /**
     * The function <i>f(x,y) = y</i>
     */
    static const std::reference_wrapper<IFXY> Y;
    /**
     * The function <i>f(x,y) = 1</i>
     */
    static const std::reference_wrapper<IFXY> ONE;

    virtual ~IFXY();
    /**
     * The function <i>z = f(x,y)</i>
     *
     * @param x the x parameter
     * @param y the y parameter
     *
     * @return z
     */
    virtual double F(double x, double y) = 0;

    /**
     * Returns the function as C code expression, e.g. <code>"pow(x, 3) * y"</code>
     * @return the C code
     */
    virtual std::string GetCCodeExpr() = 0;
};

}  // namespace snapengine
}  // namespace alus
