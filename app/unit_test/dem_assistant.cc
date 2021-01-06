
#include "dem_assistant.h"

#include <stdexcept>

#include "gmock/gmock.h"

namespace {

using ::testing::ContainerEq;
using ::testing::IsFalse;
using ::testing::IsTrue;
using ::testing::Throw;

using namespace alus::app;

TEST(DemAssistantArgumentExtract, ThrowsWhenInvalidArgumentsSupplied) {
    EXPECT_THROW(DemAssistant::ArgumentsExtract::ExtractSrtm3Files({{"srtm_41_01.tif"}, {"srtm_42_01.txt"}}),
                 std::invalid_argument);
    EXPECT_THROW(DemAssistant::ArgumentsExtract::ExtractSrtm3Files({{"srtm_41_01.tiff"}, {"srtm_42_01.tif"}}),
                 std::invalid_argument);
    EXPECT_THROW(DemAssistant::ArgumentsExtract::ExtractSrtm3Files({{"srtm_42_01.tif srtm_41_01.tiff"}}),
                 std::invalid_argument);
}

TEST(DemAssistantArgumentExtract, DetectsInvalidFilePatterns) {
    EXPECT_THAT(DemAssistant::ArgumentsExtract::IsValid("srtm_42_01.tiff"), IsFalse());
    EXPECT_THAT(DemAssistant::ArgumentsExtract::IsValid("srtm_42_01"), IsFalse());
    EXPECT_THAT(DemAssistant::ArgumentsExtract::IsValid("srtm_42_01.txt"), IsFalse());
    EXPECT_THAT(DemAssistant::ArgumentsExtract::IsValid("srtm_42_01_tif"), IsFalse());
}

TEST(DemAssistantArgumentExtract, DetectsValidFilePatterns) {
    EXPECT_THAT(DemAssistant::ArgumentsExtract::IsValid("srtm_42_01.tif"), IsTrue());
    EXPECT_THAT(DemAssistant::ArgumentsExtract::IsValid("srtm_46_04.tif"), IsTrue());
}

TEST(DemAssistan, RetainsInitialVectorValues) {
    std::vector<std::string> expected{{"/some/path/srtm42_01.tif"}, {"./path/srtm_40_05.tif"}, {"./srtm_39_17.tif"}};
    const auto& result = DemAssistant::ArgumentsExtract::ExtractSrtm3Files(expected);
    EXPECT_THAT(result, ContainerEq(expected));
}

TEST(DemAssistant, ParsesCorrectlySpaceSeparatedDemFiles) {
    std::vector<std::string> expected{{"/some/path/srtm42_01.tif"}, {"./path/srtm_40_05.tif"}, {"./srtm_39_17.tif"}};
    const auto& result =
        DemAssistant::ArgumentsExtract::ExtractSrtm3Files({"/some/path/srtm42_01.tif "
                                                           "./path/srtm_40_05.tif ./srtm_39_17.tif"});
    EXPECT_THAT(result, ContainerEq(expected));
}

TEST(DemAssistant, ParsesCorrectlyMixedSuppliedDemFiles) {
    std::vector<std::string> expected{
        {"/path/srtm_38_09.tif"}, {"srtm_41_01.tif"}, {"srtm_20_08.tif"}, {"./path/srtm_21_09.tif"}};
    const auto& result = DemAssistant::ArgumentsExtract::ExtractSrtm3Files({{"/path/srtm_38_09.tif srtm_41_01.tif "
                                                                             "srtm_20_08.tif"},
                                                                            {"./path/srtm_21_09.tif"}});
    EXPECT_THAT(result, ContainerEq(result));
}
}  // namespace