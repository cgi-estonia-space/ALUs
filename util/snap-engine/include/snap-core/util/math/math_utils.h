#pragma once

#include <cmath>

namespace alus {
namespace snapengine {

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
    static double LOG10;

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
    static short Crop(short val, short min, short max) { return val < min ? min : val > max ? max : val; }

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
    static long Crop(long val, long min, long max) { return val < min ? min : val > max ? max : val; }

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
};

double MathUtils::LOG10 = log(10.0);

}  // namespace snapengine
}  // namespace alus
