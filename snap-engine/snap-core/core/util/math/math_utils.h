/**
 * This file is a filtered duplicate of a SNAP's
 * org.esa.snap.core.util.math.MathUtils.java
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

#include <cmath>
#include <memory>
#include <vector>

namespace alus::snapengine {
namespace custom {
struct Rectangle;
struct Dimension;
}  // namespace custom

class MathUtils {
public:
    // from java lang
    static constexpr double PI = 3.14159265358979323846;
    /**
     *
     * The epsilon value for the <code>float</code> data type. The exact value of this constant is <code>1.0E-6</code>.
     */
    static constexpr float EPS_F = 1.0E-6F;

    /**
     * The epsilon value for the <code>double</code> data type. The exact value of this constant is
     * <code>1.0E-12</code>.
     */
    static constexpr double EPS = 1.0E-12;

    /**
     * Conversion factor for degrees to radians for the <code>double</code> data type.
     */
    static constexpr double DTOR = PI / 180.0;

    /**
     * Conversion factor for radians to degrees for the <code>double</code> data type.
     */
    static constexpr double RTOD = 180.0 / PI;

    /**
     * Conversion factor for degrees to radians for the <code>float</code> data type.
     */
    static constexpr float DTOR_F = static_cast<float>(DTOR);

    /**
     * Conversion factor for radians to degrees for the <code>float</code> data type.
     */
    static constexpr float RTOD_F = static_cast<float>(RTOD);

    /**
     * The natural logarithm of 10 as given by <code>Math.logging(10)</code>
     */
    static double log_10_;

    /**
     * Pi half
     */
    static constexpr double HALFPI = PI * 0.5;

    /**
     * Performs a fast linear interpolation in two dimensions i and j.
     *
     * @param wi  weight in i-direction, a weight of 0.0 corresponds to i, 1.0 to i+1
     * @param wj  weight in j-direction, a weight of 0.0 corresponds to j, 1.0 to j+1
     * @param x00 first anchor point located at (i,j)
     * @param x10 second anchor point located at (i+1,j)
     * @param x01 third anchor point located at (i,j+1)
     * @param x11 forth anchor point located at (i+1,j+1)
     *
     * @return the interpolated value
     */
    //    static float Interpolate2D(float wi, float wj, float x00, float x10, float x01, float x11);

    /**
     * Performs a fast linear interpolation in two dimensions i and j.
     *
     * @param wi  weight in i-direction, a weight of 0.0 corresponds to i, 1.0 to i+1
     * @param wj  weight in j-direction, a weight of 0.0 corresponds to j, 1.0 to j+1
     * @param x00 first anchor point located at (i,j)
     * @param x10 second anchor point located at (i+1,j)
     * @param x01 third anchor point located at (i,j+1)
     * @param x11 forth anchor point located at (i+1,j+1)
     *
     * @return the interpolated value
     */
    static double Interpolate2D(double wi, double wj, double x00, double x10, double x01, double x11);

    /**
     * First calls <code>Math.floor</code> with <code>x</code> and then crops the resulting value to the range
     * <code>min</code> to <code>max</code>.
     */
    static int FloorAndCrop(double x, int min, int max);

    /**
     * Returns <code>(int) Math.floor(value)</code>.
     *
     * @param value the <code>double</code> value to be converted
     *
     * @return the integer value corresponding to the floor of <code>value</code>
     */
    static int FloorInt(double value) { return static_cast<int>(floor(value)); }

    /**
     * Crops the value to the range <code>min</code> to <code>max</code>.
     *
     * @param val the value to crop
     * @param min the minimum crop limit
     * @param max the maximum crop limit
     */
    static int16_t Crop(int16_t val, int16_t min, int16_t max) { return val < min ? min : val > max ? max : val; }

    /**
     * Crops the value to the range <code>min</code> to <code>max</code>.
     *
     * @param val the value to crop
     * @param min the minimum crop limit
     * @param max the maximum crop limit
     */
    static int Crop(int val, int min, int max) { return val < min ? min : val > max ? max : val; }

    /**
     * Crops the value to the range <code>min</code> to <code>max</code>.
     *
     * @param val the value to crop
     * @param min the minimum crop limit
     * @param max the maximum crop limit
     */
    static int64_t Crop(int64_t val, int64_t min, int64_t max) { return val < min ? min : val > max ? max : val; }

    /**
     * Crops the value to the range <code>min</code> to <code>max</code>.
     *
     * @param val the value to crop
     * @param min the minimum crop limit
     * @param max the maximum crop limit
     */
    static float Crop(float val, float min, float max) { return val < min ? min : val > max ? max : val; }

    /**
     * Crops the value to the range <code>min</code> to <code>max</code>.
     *
     * @param val the value to crop
     * @param min the minimum crop limit
     * @param max the maximum crop limit
     */
    static double Crop(double val, double min, double max) { return val < min ? min : val > max ? max : val; }

    /**
     * Computes an integer dimension for a given integer area that best fits the
     * rectangle given by floating point width and height.
     *
     * @param n the integer area
     * @param a the rectangle's width
     * @param b the rectangle's height
     *
     * @return an integer dimension, never null
     */
    static std::shared_ptr<custom::Dimension> FitDimension(int n, double a, double b);

    /**
     * Subdivides a rectangle into tiles. The coordinates of each returned tile rectangle are guaranteed
     * to be within the given rectangle.
     *
     * @param width       the rectangle's width
     * @param height      the rectangle's height
     * @param numTilesX   the number of tiles in X direction
     * @param numTilesY   the number of tiles in Y direction
     * @param extraBorder an extra border size to extend each tile
     *
     * @return the tile coordinates as rectangles
     */
    static std::vector<std::shared_ptr<custom::Rectangle>> SubdivideRectangle(int width, int height, int num_tiles_x,
                                                                              int num_tiles_y, int extra_border);

    /**
     * Compares two double values for equality within the given epsilon.
     *
     * @param x1  the first value
     * @param x2  the second value
     * @param eps the maximum allowed difference
     */
    static bool EqualValues(double x1, double x2, double eps);
};

}  // namespace alus::snapengine