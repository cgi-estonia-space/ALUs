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

#include <cstddef>

#include "cuda_stubs.h"

namespace alus::math::polynomials {

inline DEVICE_STUB HOST_STUB double CalculateValue(double x, double* coefficients, int coefficients_length) {
    double val = 0.0;

    for (int i = coefficients_length - 1; i > 0; i--) {
        val += coefficients[i];
        val *= x;
    }

    return val + coefficients[0];
}

}  // namespace alus::math::polynomials
