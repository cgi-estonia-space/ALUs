#include <chrono>
#include <fstream>
#include <numeric>

#include "gtest/gtest.h"

#include "terrain_correction.hpp"
#include "terrain_correction_test.cuh"
#include "local_dem.cuh"

namespace {

class TerrainCorrectionIntegrationTest : public ::testing::Test {
   public:
    TerrainCorrectionIntegrationTest() = default;
};

TEST_F(TerrainCorrectionIntegrationTest, Saaremaa1) {
    alus::Dataset input(
        "goods/S1A_IW_SLC__1SDV_20190715T160437_20190715T160504_028130_032D5B_58D6_Orb_Stack_coh_deb"
        ".tif");
    alus::Dataset dem("goods/srtm_41_01.tif");
    alus::Dataset expectation("goods/S1A_IW_SLC__1SDV_20190715T160437_20190715T160504_028130_032D5B_58D6_Orb_Stack_coh_deb_TC"
                              ".tif");
    input.LoadRasterBand(1);
    expectation.LoadRasterBand(1);
    dem.LoadRasterBand(1);
    alus::TerrainCorrection alg(std::move(input), std::move(dem));

    auto const main_alg_start = std::chrono::steady_clock::now();
    alus::Dataset result = alg.ExecuteTerrainCorrection();
    auto const main_alg_stop = std::chrono::steady_clock::now();

    std::cout << "ALG spent " << std::chrono::duration_cast<std::chrono::milliseconds>(main_alg_stop - main_alg_start).count() << "ms" <<
              std::endl;

    double control_geo_transform[6], result_geo_transform[6];
    result.LoadRasterBand(1);
    expectation.GetGdalDataset()->GetGeoTransform(control_geo_transform);
    result.GetGdalDataset()->GetGeoTransform(result_geo_transform);

    for (int i = 0; i < 6; ++i) {
        EXPECT_DOUBLE_EQ(control_geo_transform[i], result_geo_transform[i]);
    }

    EXPECT_TRUE(alus::integrationtests::AreVectorsEqual(expectation.GetDataBuffer(), result.GetDataBuffer()));
}
}
