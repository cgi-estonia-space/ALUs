#include "terrain_correction.hpp"

#include <numeric>
#include <optional>

#include "gmock/gmock.h"

#include "tests_common.hpp"

using namespace alus::tests;

namespace {

class TerrainCorrectionTest : public ::testing::Test {
   public:
    TerrainCorrectionTest() {
        cohDs = std::make_optional<alus::Dataset>(COH_1_TIF);
        cohDs.value().loadRasterBand(1);
        demDs = std::make_optional<alus::Dataset>(DEM_PATH_1);
        demDs.value().loadRasterBand(1);
    }

    std::optional<alus::Dataset> cohDs;
    std::optional<alus::Dataset> cohDataDs;
    std::optional<alus::Dataset> demDs;

   protected:
};

TEST_F(TerrainCorrectionTest, fetchElevationsOnGPU) {
    alus::TerrainCorrection tc{std::move(cohDs.value()),
                               std::move(cohDs.value()),
                               std::move(demDs.value())};
    tc.localDemCuda();
    const auto& elevations = tc.getElevations();
    const auto [min, max] =
    std::minmax_element(std::begin(elevations), std::end(elevations));
    EXPECT_EQ(*min, 0);
    EXPECT_EQ(*max, 43);
    auto const avg = std::accumulate(elevations.cbegin(), elevations.cend(), 0.0) / elevations.size();
    EXPECT_DOUBLE_EQ(avg, 2.960957384655039);
}

}  // namespace
