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

#include "gdal_util.h"

#include "gmock/gmock.h"

namespace {

using ::testing::Eq;

TEST(GdalUtilIntegrationTest, ConvertsShapefileToWkt) {
    const std::string_view expected_wkt =
        "POLYGON ((4.46044921875 50.2296379178968,4.9932861328125 50.2190946204475,4.98779296875 "
        "50.0712436604447,4.449462890625 50.1029644872335,4.46044921875 50.2296379178968))";
    const auto result_wkt = alus::ConvertToWkt("./goods/alus_aoi.shp");
    EXPECT_THAT(result_wkt, Eq(expected_wkt));
}

}  // namespace