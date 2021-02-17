/**
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
inline __device__ __host__ double Interpolate2D(double i_weight, double j_weight, double anchor_1, double anchor_2,
                                                double anchor_3, double anchor_4) {
    return anchor_1 + i_weight * (anchor_2 - anchor_1) + j_weight * (anchor_3 - anchor_1) +
           i_weight * j_weight * (anchor_4 + anchor_1 - anchor_3 - anchor_2);
}

inline __device__ __host__ bool Xor(const bool& a, const bool& b) { return !a != !b; }

/**
 * Non-atomic implementation of Compare-And-Swap function. It compares value stored in pointer with some other value and
 * overwrites it with a new value.
 *
 * @tparam T Any type which supports equality operator or has it overridden.
 * @param old Pointer to the old value.
 * @param compare Value, with which the old value will be compared.
 * @param value The value to be written in the old memory address.
 * @return The old value.
 */
template <typename T>
inline __device__ __host__ T Cas(T* old, T compare, T value) {
    T old_value = *old;
    *old = (old_value == compare) * value + (old_value != compare) * old_value;

    return old_value;
}

/** Utility function that can be used to replace if..else clause by using arithmetics and bool conversion to int
 *
 * @tparam T Any numerical value
 * @param predicate Predicate based on which the returned value will be chosen.
 * @param true_case Value that should be returned when the predicate is evaluated to true.
 * @param false_case Value that should be returned when the predicate is evaluated to false.
 * @return true_case or false_case based on the predicate.
 */
template <typename T>
inline __device__ __host__ T ChooseOne(bool predicate, const T& true_case, const T& false_case) {
    return predicate * true_case + (1 - predicate) * false_case;
}

}  // namespace mathutils
}  // namespace alus