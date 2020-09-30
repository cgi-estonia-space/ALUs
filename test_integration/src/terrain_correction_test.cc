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
#include "terrain_correction.h"

#include <openssl/md5.h>
#include <boost/filesystem.hpp>
#include <boost/iostreams/device/mapped_file.hpp>

#include <chrono>
#include <cstddef>
#include <memory>
#include <numeric>
#include <vector>

#include "gmock/gmock.h"
#include "srtm3_elevation_model.h"

namespace {

using ::testing::Eq;
using ::testing::IsTrue;

using namespace alus::terraincorrection;

std::string Md5FromFile(const std::string& path) {
    unsigned char result[MD5_DIGEST_LENGTH];
    boost::iostreams::mapped_file_source src(path);
    MD5((unsigned char*)src.data(), src.size(), result);
    std::ostringstream sout;
    sout << std::hex << std::setfill('0');
    for (auto c : result) sout << std::setw(2) << (int)c;
    return sout.str();
}

class TerrainCorrectionIntegrationTest : public ::testing::Test {
public:
    TerrainCorrectionIntegrationTest() = default;
};

TEST_F(TerrainCorrectionIntegrationTest, Saaremaa1) {
    const int selected_band{1};
    std::string const coh_1_tif{
        "./goods/"
        "S1A_IW_SLC__1SDV_20190715T160437_20190715T160504_028130_032D5B_58D6_Orb_"
        "Stack_coh_deb.tif"};
    std::string const coh_1_data{
        "./goods/"
        "S1A_IW_SLC__1SDV_20190715T160437_20190715T160504_028130_032D5B_58D6_Orb_"
        "Stack_coh_deb.data"};

    Metadata metadata(coh_1_data.substr(0, coh_1_data.length() - 5) + ".dim",
                      coh_1_data + "/tie_point_grids/latitude.img", coh_1_data + "/tie_point_grids/longitude.img");
    alus::Dataset<double> input(coh_1_tif);

    auto egm_96 = std::make_shared<alus::snapengine::EarthGravitationalModel96>();
    egm_96->HostToDevice();

    std::vector<std::string> files{"./goods/srtm_41_01.tif", "./goods/srtm_42_01.tif"};
    auto srtm_3_model = std::make_unique<alus::snapengine::Srtm3ElevationModel>(files);
    srtm_3_model->ReadSrtmTiles(egm_96.get());
    srtm_3_model->HostToDevice();

    const auto* d_srtm_3_tiles = srtm_3_model->GetSrtmBuffersInfo();
    const size_t srtm_3_tiles_length{2};

    const std::string output_path{"/tmp/tc_test.tif"};
    auto const main_alg_start = std::chrono::steady_clock::now();

    TerrainCorrection tc(std::move(input), metadata.GetMetadata(), metadata.GetLatTiePoints(),
                         metadata.GetLonTiePoints(), d_srtm_3_tiles, srtm_3_tiles_length, selected_band);
    tc.ExecuteTerrainCorrection(output_path, 420, 416);

    auto const main_alg_stop = std::chrono::steady_clock::now();
    std::cout << "ALG spent "
              << std::chrono::duration_cast<std::chrono::milliseconds>(main_alg_stop - main_alg_start).count() << "ms"
              << std::endl;

    ASSERT_THAT(boost::filesystem::exists(output_path), IsTrue());
    const std::string expected_md5{"aa72aab946bb25eb35eee58085254cff"};
    ASSERT_THAT(Md5FromFile(output_path), Eq(expected_md5));
}
}  // namespace
