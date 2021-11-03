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

#include "../../snap-engine/geo_utils.h"

#include <vector>

#include "gmock/gmock.h"

namespace {

using namespace alus;
using namespace alus::snapengine;
using namespace alus::snapengine::geoutils;

class GeoUtilsTest : public ::testing::Test {};

TEST_F(GeoUtilsTest, geo2xyzWGS84ComputesCorrectly) {
    // Test data from getSourceRectangle(), no data case of a loop and a regular loop from RangeDopplerGeocodingOp.java.
    // Copied by running compiled SNAP's Java code.
    std::vector<double> const lats{58.53737986139701, 58.56337063784938, 58.56099648038498};
    std::vector<double> const lons{22.210274275974594, 22.28899633926784, 23.008615962292925};
    std::vector<double> const alts{44.81804549607166, 46.58281173501725, 28.723447630795434};

    std::vector<PosVector> const expectedResults{{3089569.9005578943, 1261476.4569731771, 5417255.575817931},
                                                 {3085549.3787867613, 1264783.8242229724, 5418767.576867359},
                                                 {3069620.195372688, 1303521.273610704, 5418614.4075967185}};

    // Input data sanity check.
    ASSERT_EQ(lats.size(), lons.size());
    ASSERT_EQ(lats.size(), alts.size());
    ASSERT_EQ(expectedResults.size(), lats.size());

    for (size_t i = 0; i < lats.size(); i++) {
        PosVector pos{};
        Geo2xyzWgs84(lats.at(i), lons.at(i), alts.at(i), pos);
        auto const expected = expectedResults.at(i);
        EXPECT_DOUBLE_EQ(expected.x, pos.x);
        EXPECT_DOUBLE_EQ(expected.y, pos.y);
        EXPECT_DOUBLE_EQ(expected.z, pos.z);
    }
}

}  // namespace