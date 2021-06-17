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

#include <array>
#include <cstddef>
#include <memory>
#include <numeric>
#include <string_view>
#include <vector>

#include <boost/container_hash/hash.hpp>
#include <boost/filesystem.hpp>
#include <boost/iostreams/device/mapped_file.hpp>
#include <gdal.h>
#include <openssl/md5.h>

#include "gmock/gmock.h"

#include "gdal_util.h"
#include "srtm3_elevation_model.h"

namespace {

using ::testing::Eq;
using ::testing::IsTrue;

using namespace alus;
using namespace alus::terraincorrection;

std::string Md5FromFile(const std::string& path) {
    unsigned char result[MD5_DIGEST_LENGTH];
    boost::iostreams::mapped_file_source src(path);
    MD5(reinterpret_cast<const unsigned char*>(src.data()), src.size(), result);
    std::ostringstream sout;
    sout << std::hex << std::setfill('0');
    for (auto c : result) sout << std::setw(2) << static_cast<int>(c);
    return sout.str();
}

std::string HashFromBand(std::string_view file_path) {
    auto* const dataset = static_cast<GDALDataset*>(GDALOpen(file_path.data(), GA_ReadOnly));
    const auto x_size = dataset->GetRasterXSize();
    const auto y_size = dataset->GetRasterYSize();

    std::vector<float> raster_data(x_size * y_size);
    auto error = dataset->GetRasterBand(1)->RasterIO(GF_Read, 0, 0, x_size, y_size, raster_data.data(), x_size, y_size,
                                                     GDALDataType::GDT_Float32, 0, 0);
    CHECK_GDAL_ERROR(error);
    GDALClose(dataset);

    std::ostringstream s_out;
    s_out << std::hex << std::setfill('0')
          << boost::hash<std::vector<float>>{}(raster_data);

    return s_out.str();
}

class TerrainCorrectionIntegrationTest : public ::testing::Test {
public:
    TerrainCorrectionIntegrationTest() = default;

protected:
    void CompareGeocoding(std::string_view reference_file, std::string_view comparand_file) {
        auto* const reference_dataset =
            static_cast<GDALDataset*>(GDALOpen(std::string(reference_file).c_str(), GA_ReadOnly));
        auto* const comparand_dataset =
            static_cast<GDALDataset*>(GDALOpen(std::string(comparand_file).c_str(), GA_ReadOnly));

        const int geo_transform_length{6};
        std::array<double, geo_transform_length> reference_geo_transform;
        std::array<double, geo_transform_length> comparand_geo_transform;
        reference_dataset->GetGeoTransform(reference_geo_transform.data());
        comparand_dataset->GetGeoTransform(comparand_geo_transform.data());
        GDALClose(reference_dataset);
        GDALClose(comparand_dataset);

        ASSERT_THAT(comparand_geo_transform,
                    ::testing::ElementsAreArray(reference_geo_transform.begin(), reference_geo_transform.end()));
    }
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
    Dataset<double> input(coh_1_tif);

    auto egm_96 = std::make_shared<snapengine::EarthGravitationalModel96>();
    egm_96->HostToDevice();

    std::vector<std::string> files{"./goods/srtm_41_01.tif", "./goods/srtm_42_01.tif"};
    auto srtm_3_model = std::make_unique<snapengine::Srtm3ElevationModel>(files);
    srtm_3_model->ReadSrtmTiles(egm_96.get());
    srtm_3_model->HostToDevice();

    const auto* d_srtm_3_tiles = srtm_3_model->GetSrtmBuffersInfo();
    const size_t srtm_3_tiles_length{2};

    const std::string output_path{"/tmp/tc_test.tif"};

    TerrainCorrection tc(std::move(input), metadata.GetMetadata(), metadata.GetLatTiePointGrid(),
                         metadata.GetLonTiePointGrid(), d_srtm_3_tiles, srtm_3_tiles_length, selected_band);
    tc.ExecuteTerrainCorrection(output_path, 420, 416);

    ASSERT_THAT(boost::filesystem::exists(output_path), IsTrue());
    const std::string expected_md5{"67458d461c814e4b00f894956c08285a"};
    ASSERT_THAT(Md5FromFile(output_path), Eq(expected_md5));

    CompareGeocoding(
        "./goods/"
        "S1A_IW_SLC__1SDV_20190715T160437_20190715T160504_028130_032D5B_58D6_Orb_Stack_coh_deb_TC.tif",
        output_path);
}

TEST_F(TerrainCorrectionIntegrationTest, BeirutExplosion) {
    const int selected_band{1};
    std::string const coh_1_tif{
        "./goods/terrain_correction/"
        "Beirut_IW1_6_VH_orb_stack_cor_deb_coh.tif"};
    std::string const coh_1_data{
        "./goods/terrain_correction/"
        "Beirut_IW1_6_VH_orb_stack_cor_deb_coh.data"};

    Metadata metadata(coh_1_data.substr(0, coh_1_data.length() - 5) + ".dim",
                      coh_1_data + "/tie_point_grids/latitude.img", coh_1_data + "/tie_point_grids/longitude.img");
    Dataset<double> input(coh_1_tif);

    auto egm_96 = std::make_shared<snapengine::EarthGravitationalModel96>();
    egm_96->HostToDevice();

    std::vector<std::string> files{"./goods/srtm_43_06.tif", "./goods/srtm_44_06.tif"};
    auto srtm_3_model = std::make_unique<snapengine::Srtm3ElevationModel>(files);
    srtm_3_model->ReadSrtmTiles(egm_96.get());
    srtm_3_model->HostToDevice();

    const auto* d_srtm_3_tiles = srtm_3_model->GetSrtmBuffersInfo();
    const size_t srtm_3_tiles_length{2};

    const std::string output_path{"/tmp/tc_beirut_test.tif"};

    TerrainCorrection tc(std::move(input), metadata.GetMetadata(), metadata.GetLatTiePointGrid(),
                         metadata.GetLonTiePointGrid(), d_srtm_3_tiles, srtm_3_tiles_length, selected_band);
    tc.ExecuteTerrainCorrection(output_path, 420, 416);

    ASSERT_THAT(boost::filesystem::exists(output_path), IsTrue());
    CompareGeocoding(
        "./goods/terrain_correction/"
        "Beirut_IW1_6_VH_orb_stack_cor_deb_coh_TC.tif",
        output_path);

    const std::string expected_boost_hash{"fa952a77788339ee"};
    ASSERT_THAT(HashFromBand(output_path), ::testing::Eq(expected_boost_hash));
}
}  // namespace
