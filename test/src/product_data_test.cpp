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

#include "gmock/gmock.h"

#include "../goods/S1A_IW_SLC__1SDV_20190715T160437_20190715T160504_028130_032D5B_58D6_Orb_Stack_coh_deb_orbit.hpp"

namespace {

using namespace alus::goods;
using namespace alus::snapengine;

TEST(ProductData, UtcHasCorrectMjdTimes) {
    auto const series_size = ORBIT_STATE_VECTORS.size();
    EXPECT_EQ(series_size, MJD_TIMES.size());
    for (size_t i = 0; i < series_size; i++) {
        EXPECT_EQ(ORBIT_STATE_VECTORS.at(i).timeMjd_, MJD_TIMES.at(i));
    }
}
TEST(ProductData, UtcIsCorrectlyConstructedFromString) {
    const std::string DATE_TEXT = "15-JUL-2019 16:04:43.800577";
    const alus::snapengine::old::Utc EXPECTED_UTC{7135, 57883, 800577};
    const alus::snapengine::old::Utc RESULT(DATE_TEXT);
    EXPECT_EQ(EXPECTED_UTC.days_, RESULT.days_);
    EXPECT_EQ(EXPECTED_UTC.seconds_, RESULT.seconds_);
    EXPECT_EQ(EXPECTED_UTC.microseconds_, RESULT.microseconds_);
}
}  // namespace