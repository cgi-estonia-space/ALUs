#include "terrain_correction.hpp"

#include "gmock/gmock.h"

namespace {

std::string testFile{"/home/sven/snap-products/S1A_IW_SLC__1SDV_20190715T160437_20190715T160504_028130_032D5B_58D6_Orb_Stack_coh_deb.tif"};

class TerrainCorrectionTest : public ::testing::Test {
   public:
    TerrainCorrectionTest() : ds{std::make_shared<slap::Dataset>(testFile)} {}

   protected:
    std::shared_ptr<slap::Dataset> ds;
};

TEST_F(TerrainCorrectionTest, doSomeWork) {
    slap::TerrainCorrection tc{ds};
    tc.doWork();
}
}  // namespace