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
#include <chrono>
#include <numeric>

#include <openssl/md5.h>
#include <boost/filesystem.hpp>
#include <boost/iostreams/device/mapped_file.hpp>

#include "gmock/gmock.h"

#include "terrain_correction.h"

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
    std::string const COH_1_TIF{
        "./goods/"
        "S1A_IW_SLC__1SDV_20190715T160437_20190715T160504_028130_032D5B_58D6_Orb_"
        "Stack_coh_deb.tif"};
    std::string const COH_1_DATA{
        "./goods/"
        "S1A_IW_SLC__1SDV_20190715T160437_20190715T160504_028130_032D5B_58D6_Orb_"
        "Stack_coh_deb.data"};

    Metadata metadata(COH_1_DATA.substr(0, COH_1_DATA.length() - 5) + ".dim",
                      COH_1_DATA + "/tie_point_grids/latitude.img",
                      COH_1_DATA + "/tie_point_grids/longitude.img");
    alus::Dataset input(COH_1_TIF);

    const std::string output_path{"/tmp/tc_test.tif"};
    auto const main_alg_start = std::chrono::steady_clock::now();
    TerrainCorrection tc(
        std::move(input), metadata.GetMetadata(), metadata.GetLatTiePoints(), metadata.GetLonTiePoints());
    tc.ExecuteTerrainCorrection(output_path, 420, 416);

    auto const main_alg_stop = std::chrono::steady_clock::now();
    std::cout << "ALG spent "
              << std::chrono::duration_cast<std::chrono::milliseconds>(main_alg_stop - main_alg_start).count() << "ms"
              << std::endl;

    ASSERT_THAT(boost::filesystem::exists(output_path), IsTrue());
    const std::string expected_md5{"1234bfe807838b6901f837860453f71a"};
    ASSERT_THAT(Md5FromFile(output_path), Eq(expected_md5));
}
}  // namespace
