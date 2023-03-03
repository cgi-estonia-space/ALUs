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

TEST(AssistantArgumentExtract, ThrowsWhenInvalidSrtm3ArgumentsSupplied) {
    EXPECT_THROW(Assistant::ArgumentsExtract::ExtractSrtm3Files({{"srtm_41_01.tif"}, {"srtm_42_01.txt"}}),
                 std::invalid_argument);
    EXPECT_THROW(Assistant::ArgumentsExtract::ExtractSrtm3Files({{"srtm_41_01.tiff"}, {"srtm_42_01.tif"}}),
                 std::invalid_argument);
    EXPECT_THROW(Assistant::ArgumentsExtract::ExtractSrtm3Files({{"srtm_42_01.tif srtm_41_01.tiff"}}),
                 std::invalid_argument);
}

TEST(AssistantArgumentExtract, DetectsInvalidSrtm3FilePatterns) {
    EXPECT_THAT(Assistant::ArgumentsExtract::IsValidSrtm3Filename("srtm_42_01.tiff"), IsFalse());
    EXPECT_THAT(Assistant::ArgumentsExtract::IsValidSrtm3Filename("srtm_42_01"), IsFalse());
    EXPECT_THAT(Assistant::ArgumentsExtract::IsValidSrtm3Filename("srtm_42_01.txt"), IsFalse());
    EXPECT_THAT(Assistant::ArgumentsExtract::IsValidSrtm3Filename("srtm_42_01_tif"), IsFalse());
}

TEST(AssistantArgumentExtract, DetectsValidSrtm3FilePatterns) {
    EXPECT_THAT(Assistant::ArgumentsExtract::IsValidSrtm3Filename("srtm_42_01.tif"), IsTrue());
    EXPECT_THAT(Assistant::ArgumentsExtract::IsValidSrtm3Filename("srtm_46_04.tif"), IsTrue());
}

TEST(AssistantArgumentExtract, DetectsValidCopDemCog30mFilePatterns) {
    EXPECT_THAT(
        Assistant::ArgumentsExtract::IsValidCopDemCog30mFilename({"Copernicus_DSM_COG_10_N49_00_E005_00_DEM.tif"}),
        IsTrue());
    EXPECT_THAT(Assistant::ArgumentsExtract::IsValidCopDemCog30mFilename(
                    {"/tmp/folder/Copernicus_DSM_COG_10_N49_00_E005_00_DEM.tif"}), IsTrue());
}

TEST(AssistantArgumentExtract, DetectsInvalidCopDemCog30mFilePatterns) {
    EXPECT_THAT(Assistant::ArgumentsExtract::IsValidCopDemCog30mFilename({"Copernicus_DSM_10_S18_00_E020_00_DEM.tif"}),
                IsFalse());
    EXPECT_THAT(Assistant::ArgumentsExtract::IsValidCopDemCog30mFilename({"Copernicus_DSM_S18_00_E020_00_DEM.tif"}),
                IsFalse());
    EXPECT_THAT(
        Assistant::ArgumentsExtract::IsValidCopDemCog30mFilename({"Copernicus_DSS_COG_10_N49_00_E005_00_DEM.tif"}),
        IsFalse());
}

TEST(Assistant, RetainsInitialSrtm3VectorValues) {
    std::vector<std::string> expected{{"/some/path/srtm_42_01.tif"}, {"./path/srtm_40_05.tif"}, {"./srtm_39_17.tif"}};
    const auto& result = Assistant::ArgumentsExtract::ExtractSrtm3Files(expected);
    EXPECT_THAT(result, ContainerEq(expected));
}

TEST(Assistant, ParsesCorrectlySpaceSeparatedSrtm3DemFiles) {
    std::vector<std::string> expected{{"/some/path/srtm_42_01.tif"}, {"./path/srtm_40_05.tif"}, {"./srtm_39_17.tif"}};
    const auto& result =
        Assistant::ArgumentsExtract::ExtractSrtm3Files({"/some/path/srtm_42_01.tif "
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

TEST(Assistant, PreparesCmdArgInputCorrectly) {
    const std::vector<std::string> example1{"srtm_41_01.tif srtm_20_08.tif    ./path/srtm_21_09.tif"};
    const std::vector<std::string> expected1{{"srtm_41_01.tif"}, {"srtm_20_08.tif"}, {"./path/srtm_21_09.tif"}};
    const auto result1 = Assistant::ArgumentsExtract::PrepareArgs(example1);
    ASSERT_THAT(result1, ContainerEq(expected1));

    const std::vector<std::string> example2{
        "Copernicus_DSM_COG_10_N49_00_E005_00_DEM.tif", "Copernicus_DSM_COG_10_N49_00_E009_00_DEM.tif",
        "Copernicus_DSM_COG_10_N50_00_E005_00_DEM.tif Copernicus_DSM_COG_10_N51_00_E005_00_DEM.tif"};
    const std::vector<std::string> expected2{
        "Copernicus_DSM_COG_10_N49_00_E005_00_DEM.tif", "Copernicus_DSM_COG_10_N49_00_E009_00_DEM.tif",
        "Copernicus_DSM_COG_10_N50_00_E005_00_DEM.tif", "Copernicus_DSM_COG_10_N51_00_E005_00_DEM.tif"};
    const auto result2 = Assistant::ArgumentsExtract::PrepareArgs(example2);
    ASSERT_THAT(result2, ContainerEq(expected2));
}

}  // namespace