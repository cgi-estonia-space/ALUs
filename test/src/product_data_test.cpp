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
    auto const seriesSize = ORBIT_STATE_VECTORS.size();
    EXPECT_EQ(seriesSize, MJD_TIMES.size());
    for (size_t i = 0; i < seriesSize; i++){
        EXPECT_EQ(ORBIT_STATE_VECTORS.at(i).timeMjd_, MJD_TIMES.at(i));
    }
}
}