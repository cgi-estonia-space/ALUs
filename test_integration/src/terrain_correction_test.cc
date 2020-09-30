#include <chrono>
#include <fstream>
#include <numeric>

#include "gtest/gtest.h"

#include "local_dem.cuh"
#include "product_data.h"
#include "terrain_correction.hpp"

namespace {

class TerrainCorrectionIntegrationTest : public ::testing::Test {
   public:
    TerrainCorrectionIntegrationTest() = default;
};

void ReadOrbitStateVectors(const char *file_name, alus::TerrainCorrection &terrain_correction) {
    std::ifstream data_stream{file_name};
    if (!data_stream.is_open()) {
        throw std::ios::failure("Range Doppler Terrain Correction test data file not open.");
    }
    int test_data_size;
    data_stream >> test_data_size;
    terrain_correction.metadata_.orbit_state_vectors.clear();

    for (int i = 0; i < test_data_size; i++) {
        std::string utc_string1;
        std::string utc_string2;
        double x_pos, y_pos, z_pos, x_vel, y_vel, z_vel;
        data_stream >> utc_string1 >> utc_string2 >> x_pos >> y_pos >> z_pos >> x_vel >> y_vel >> z_vel;
        utc_string1.append(" ");
        utc_string1.append(utc_string2);
        terrain_correction.metadata_.orbit_state_vectors.emplace_back(
            alus::snapengine::old::Utc(utc_string1), x_pos, y_pos, z_pos, x_vel, y_vel, z_vel);
    }

    data_stream.close();
}

TEST_F(TerrainCorrectionIntegrationTest, Saaremaa1) {
    // TODO: change file locations
    alus::Dataset input(
        "goods/S1A_IW_SLC__1SDV_20190715T160437_20190715T160504_028130_032D5B_58D6_Orb_Stack_coh_deb"
        ".tif");
    alus::Dataset dem("goods/srtm_41_01.tif");
    alus::Dataset expectation(
        "goods/S1A_IW_SLC__1SDV_20190715T160437_20190715T160504_028130_032D5B_58D6_Orb_Stack_coh_deb_TC"
        ".tif");

    expectation.LoadRasterBand(1);
    alus::TerrainCorrection alg(std::move(input), std::move(dem));

    ReadOrbitStateVectors("goods/orbit_state_vectors.txt", alg);
    auto const main_alg_start = std::chrono::steady_clock::now();
    alg.ExecuteTerrainCorrection("goods/tc_output.tif", 420, 416);
    auto const main_alg_stop = std::chrono::steady_clock::now();
    //
    std::cout << "ALG spent "
              << std::chrono::duration_cast<std::chrono::milliseconds>(main_alg_stop - main_alg_start).count() << "ms"
              << std::endl;
}
}  // namespace
