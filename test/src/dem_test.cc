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
#include "dem.h"

#include <algorithm>
#include <optional>
#include <string>

#include "gmock/gmock.h"

#include "tests_common.h"

namespace {

using alus::tests::DEM_PATH_1;
using alus::tests::TIF_PATH_1;

class DemTest : public ::testing::Test {
public:
    std::optional<alus::Dataset<double>> demDataset;

    DemTest() {
        demDataset = std::make_optional<alus::Dataset<double>>(DEM_PATH_1);
        demDataset.value().LoadRasterBand(1);
    }
};

TEST_F(DemTest, getLocalDem) {
    alus::Dem dem{std::move(demDataset.value())};
    alus::Dataset<double> ds{TIF_PATH_1};
    ds.LoadRasterBand(1);

    const auto width{ds.GetGdalDataset()->GetRasterXSize()};
    const auto height{ds.GetGdalDataset()->GetRasterYSize()};
    auto const result = dem.GetLocalDemFor(ds, 0, 0, width, height);
    ASSERT_EQ(result.size(), width * height);

    // Saaremaa is not that mountainous :)
    const auto [min, max] = std::minmax_element(begin(result), end(result));
    EXPECT_GT(*min, 20);
    EXPECT_LT(*max, 60);
    EXPECT_GT(*max, 30);
}
}  // namespace
