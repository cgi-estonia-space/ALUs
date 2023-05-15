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

#include "interpolations.h"

#include "gmock/gmock.h"

namespace {

using ::testing::DoubleNear;

TEST(Polynomials, CalculateValue) {
    constexpr size_t TEST_SERIES_LENGTH{9};
    constexpr std::array<double, TEST_SERIES_LENGTH> a{
        799820.7760572317,      0.503504268315258,     5.342649278666865E-7,
        -3.383795661580446E-13, 3.289177769023392E-20, 2.083812781936989E-25,
        -2.450770556969537E-31, 1.098328719336332E-37, 1.842434173793046E-45};
    constexpr std::array<double, TEST_SERIES_LENGTH> b{
        799820.7760577537,      0.5034602605682741,    5.342944614704603E-7,
        -3.383690165169035E-13, 3.285302031191257E-20, 2.083665499340777E-25,
        -2.448362235512942E-31, 1.092520119544095E-37, 2.384071025435546E-45};
    constexpr std::array<double, TEST_SERIES_LENGTH> expected{
        799820.7760576833,       0.5034661926064123,     5.342904804790224E-7,
        -3.383704385591093E-13,  3.285824462495056E-20,  2.083685352344798E-25,
        -2.4486868659634604E-31, 1.0933030915855847E-37, 2.311060915142981E-45};
    auto weight{0.8652046845203314};
    for (size_t i{0}; i < TEST_SERIES_LENGTH; i++) {
        ASSERT_THAT(alus::math::interpolations::Linear(a.at(i), b.at(i), weight), DoubleNear(expected.at(i), 1e-20))
            << "at element " << i;
    }

    ASSERT_THAT(alus::math::interpolations::Linear(2.0, 4.0, 0.5), DoubleNear(3.0, 1e-20));
}

}  // namespace
