#pragma once

#include <cmath>

namespace alus {
namespace mathutils {

/**
 * Utility method ported from org.esa.snap.core.util-math.MathUtils
 * First rounds down <code>x</code> and then crops the resulting value to the range
 * <code>min</code> to <code>max</code>.
 */
inline __device__ __host__ int FloorAndCrop(const double x, const int min, const int max) {
    int val = int(std::floor(x));
    return val < min ? min : val > max ? max : val;
}

/**
 * Utility method ported from org.esa.snap.core.util-math.MathUtils
 * Performs a fast linear interpolation in two dimensions i and j.
 *
 * @param i_weight  weight in i-direction, a weight of 0.0 corresponds to i, 1.0 to i+1
 * @param j_weight  weight in j-direction, a weight of 0.0 corresponds to j, 1.0 to j+1
 * @param anchor_1 first anchor point located at (i,j)
 * @param anchor_2 second anchor point located at (i+1,j)
 * @param anchor_3 third anchor point located at (i,j+1)
 * @param anchor_4 forth anchor point located at (i+1,j+1)
 *
 * @return the interpolated value
 */
inline __device__ __host__ double Interpolate2D(
    double i_weight, double j_weight, double anchor_1, double anchor_2, double anchor_3, double anchor_4) {
    return anchor_1 + i_weight * (anchor_2 - anchor_1) + j_weight * (anchor_3 - anchor_1) +
           i_weight * j_weight * (anchor_4 + anchor_1 - anchor_3 - anchor_2);
}

}  // namespace mathutils
}  // namespace alus