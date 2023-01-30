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
#include "dem_assistant.h"

#include <stdexcept>

#include "gmock/gmock.h"

namespace {

using ::testing::ContainerEq;
using ::testing::IsFalse;
using ::testing::IsTrue;

using alus::dem::Assistant;

TEST(DemAssistantArgumentExtract, ThrowsWhenInvalidSrtm3ArgumentsSupplied) {
    EXPECT_THROW(Assistant::ArgumentsExtract::ExtractSrtm3Files({{"srtm_41_01.tif"}, {"srtm_42_01.txt"}}),
                 std::invalid_argument);
    EXPECT_THROW(Assistant::ArgumentsExtract::ExtractSrtm3Files({{"srtm_41_01.tiff"}, {"srtm_42_01.tif"}}),
                 std::invalid_argument);
    EXPECT_THROW(Assistant::ArgumentsExtract::ExtractSrtm3Files({{"srtm_42_01.tif srtm_41_01.tiff"}}),
                 std::invalid_argument);
}

TEST(DemAssistantArgumentExtract, ThrowsWhenInvalidCopDem30mArgumentsSupplied) {
    EXPECT_THROW(Assistant::ArgumentsExtract::ExtractCopDem30mFiles(
                     {"Copernicus_DSM_30_S18_00_E020_00_DEM.tif Copernicus_DEM_COG_10_N49_00_E005_00_DEM.tif "
                      "Nopernicus_DSM_COG_10_N49_00_E005_00_DEM.tif"}),
                 std::invalid_argument);
    EXPECT_THROW(Assistant::ArgumentsExtract::ExtractCopDem30mFiles(
                     {"Copernicus_DSM_10_S00_00_E020_00_DEM.tif Copernicus_DSM_COG_10_N90_00_W000_00_DEM.tif "
                      "Copernicus_DSM_COG_10_N49_00_E180_00_DEM.tif"}),
                 std::invalid_argument);
}

TEST(DemAssistantArgumentExtract, DetectsInvalidSrtm3FilePatterns) {
    EXPECT_THAT(Assistant::ArgumentsExtract::IsValidSrtm3Filename("srtm_42_01.tiff"), IsFalse());
    EXPECT_THAT(Assistant::ArgumentsExtract::IsValidSrtm3Filename("srtm_42_01"), IsFalse());
    EXPECT_THAT(Assistant::ArgumentsExtract::IsValidSrtm3Filename("srtm_42_01.txt"), IsFalse());
    EXPECT_THAT(Assistant::ArgumentsExtract::IsValidSrtm3Filename("srtm_42_01_tif"), IsFalse());
}

TEST(DemAssistantArgumentExtract, DetectsValidSrtm3FilePatterns) {
    EXPECT_THAT(Assistant::ArgumentsExtract::IsValidSrtm3Filename("srtm_42_01.tif"), IsTrue());
    EXPECT_THAT(Assistant::ArgumentsExtract::IsValidSrtm3Filename("srtm_46_04.tif"), IsTrue());
}

TEST(DemAssistantArgumentExtract, DetectsValidCopDem30mFilePatterns) {
    EXPECT_THAT(Assistant::ArgumentsExtract::IsValidCopDem30mFilename({"Copernicus_DSM_10_S18_00_E020_00_DEM.tif"}),
                IsTrue());
    EXPECT_THAT(
        Assistant::ArgumentsExtract::IsValidCopDem30mFilename({"Copernicus_DSM_COG_10_N49_00_E005_00_DEM.tif"}),
        IsTrue());
}

TEST(DemAssistan, RetainsInitialSrtm3VectorValues) {
    std::vector<std::string> expected{{"/some/path/srtm42_01.tif"}, {"./path/srtm_40_05.tif"}, {"./srtm_39_17.tif"}};
    const auto& result = Assistant::ArgumentsExtract::ExtractSrtm3Files(expected);
    EXPECT_THAT(result, ContainerEq(expected));
}

TEST(Assistant, ParsesCorrectlySpaceSeparatedSrtm3DemFiles) {
    std::vector<std::string> expected{{"/some/path/srtm42_01.tif"}, {"./path/srtm_40_05.tif"}, {"./srtm_39_17.tif"}};
    const auto& result =
        Assistant::ArgumentsExtract::ExtractSrtm3Files({"/some/path/srtm42_01.tif "
                                                        "./path/srtm_40_05.tif ./srtm_39_17.tif"});
    EXPECT_THAT(result, ContainerEq(expected));
}

TEST(Assistant, ParsesCorrectlyMixedSuppliedSrtm3DemFiles) {
    std::vector<std::string> expected{
        {"/path/srtm_38_09.tif"}, {"srtm_41_01.tif"}, {"srtm_20_08.tif"}, {"./path/srtm_21_09.tif"}};
    const auto& result = Assistant::ArgumentsExtract::ExtractSrtm3Files({{"/path/srtm_38_09.tif srtm_41_01.tif "
                                                                          "srtm_20_08.tif"},
                                                                         {"./path/srtm_21_09.tif"}});
    EXPECT_THAT(result, ContainerEq(result));
}
}  // namespace