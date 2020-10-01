#include <chrono>
#include <numeric>

#include "gtest/gtest.h"

#include "terrain_correction.h"

namespace {

using namespace alus::terraincorrection;

class TerrainCorrectionIntegrationTest : public ::testing::Test {
   public:
    TerrainCorrectionIntegrationTest() = default;
};

TEST_F(TerrainCorrectionIntegrationTest, Saaremaa1) {

    std::string const COH_1_TIF{"./goods/"
        "S1A_IW_SLC__1SDV_20190715T160437_20190715T160504_028130_032D5B_58D6_Orb_"
        "Stack_coh_deb.tif"};
    std::string const COH_1_DATA{"./goods/"
        "S1A_IW_SLC__1SDV_20190715T160437_20190715T160504_028130_032D5B_58D6_Orb_"
        "Stack_coh_deb.data"};

    Metadata metadata(COH_1_DATA.substr(0, COH_1_DATA.length() -5) + ".dim",
                      COH_1_DATA + "/tie_point_grids/latitude.img",
                      COH_1_DATA + "/tie_point_grids/longitude.img");
    alus::Dataset input(COH_1_TIF);

    auto const main_alg_start = std::chrono::steady_clock::now();
    TerrainCorrection tc(
        std::move(input), metadata.GetMetadata(), metadata.GetLatTiePoints(), metadata.GetLonTiePoints());
    tc.ExecuteTerrainCorrection("/tmp/tc_test.tif", 500, 500);

    auto const main_alg_stop = std::chrono::steady_clock::now();
    std::cout << "ALG spent "
              << std::chrono::duration_cast<std::chrono::milliseconds>(main_alg_stop - main_alg_start).count() << "ms"
              << std::endl;
}
}  // namespace
