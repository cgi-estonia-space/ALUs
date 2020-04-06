#include "terrain_correction.hpp"

#include <optional>

#include "gmock/gmock.h"

#include "tests_common.hpp"

using namespace slap::tests;

namespace {

class TerrainCorrectionTest : public ::testing::Test {
   public:
    TerrainCorrectionTest() {
        cohDs = std::make_optional<slap::Dataset>(COH_1_TIF);
        demDs = std::make_optional<slap::Dataset>(DEM_PATH_1);
    }

    std::optional<slap::Dataset> cohDs;
    std::optional<slap::Dataset> cohDataDs;
    std::optional<slap::Dataset> demDs;

   protected:
};

TEST_F(TerrainCorrectionTest, doSomeWork) {
    slap::TerrainCorrection tc{std::move(cohDs.value()),
                               std::move(cohDs.value()),
                               std::move(demDs.value())};
    tc.doWork();
}
}  // namespace