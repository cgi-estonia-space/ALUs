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

#include "polynomials.h"

#include "gmock/gmock.h"

namespace {

using ::testing::DoubleNear;

TEST(Polynomials, CalculateValue) {
    std::vector<double> coefficients{799820.7760576343,      0.5034703317766596,     5.342877026814206E-7,
                                     -3.38371430810786E-13,  3.286188996919564E-20,  2.0836992050816896E-25,
                                     -2.448913381820372E-31, 1.0938494222906452E-37, 2.2601169951572742E-45};
        auto x{0.0};
        ASSERT_THAT(alus::math::polynomials::CalculateValue(x, coefficients.data(), coefficients.size()),
                    DoubleNear(799820.7760576343, 1e-20));
        x = 259000.0;
        ASSERT_THAT(alus::math::polynomials::CalculateValue(x, coefficients.data(), coefficients.size()),
                    DoubleNear(960506.6845994466, 1e-20));
}

}  // namespace