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
#include <gmock/gmock.h>

#include <vector>

#include "sentinel1_calibrate_kernel_utils.h"

namespace {

TEST(Sentinel1CalibrateUtilsTest, GetCalibrationVectorIndexBinary) {
    const std::vector<int> lines{0,     486,   973,   1619,  2106,  2592,  3236,  3722,  4209, 4859,
                                 5346,  5832,  6483,  6969,  7611,  8097,  8584,  9231,  9718, 10204,
                                 10854, 11340, 11827, 12468, 12955, 13602, 14089, 14575, 1522, 15708};
    const int expected_index_1{2};
    const int query_1{1000};

    const auto result_1 = alus::sentinel1calibrate::GetCalibrationVectorIndex(query_1, lines.size(), lines.data());
    ASSERT_THAT(result_1, ::testing::Eq(expected_index_1));

    const int expected_index_2{1};
    const int query_2{486};
    const auto result_2 = alus::sentinel1calibrate::GetCalibrationVectorIndex(query_2, lines.size(), lines.data());
    ASSERT_THAT(result_2, ::testing::Eq(expected_index_2));

    const int query_3{15708};
    const auto result_3 = alus::sentinel1calibrate::GetCalibrationVectorIndex(query_3, lines.size(), lines.data());
    ASSERT_THAT(result_3, ::testing::Eq(-1));
}
}  // namespace
