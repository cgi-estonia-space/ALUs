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
#include <string_view>
#include <vector>

#include <gdal.h>
#include <boost/filesystem.hpp>

#include "gmock/gmock.h"

#include "srtm3_elevation_model.h"
#include "test_utils.h"

namespace {

using alus::Dataset;
using alus::snapengine::EarthGravitationalModel96;
using alus::snapengine::Srtm3ElevationModel;
using alus::terraincorrection::Metadata;
using alus::terraincorrection::TerrainCorrection;

using ::testing::Eq;
using ::testing::IsTrue;

class TerrainCorrectionIntegrationTest : public ::testing::Test {
public:
    TerrainCorrectionIntegrationTest() = default;

protected:
    static void CompareGeocoding(std::string_view reference_file, std::string_view comparand_file) {
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
        "./goods/terrain_correction/"
        "S1A_IW_SLC__1SDV_20190715T160437_20190715T160504_028130_032D5B_58D6_Orb_Stack_coh_deb_data/"
        "S1A_IW_SLC__1SDV_20190715T160437_20190715T160504_028130_032D5B_58D6_Orb_Stack_coh_deb.tif"};
    std::string const coh_1_data{
        "./goods/terrain_correction/"
        "S1A_IW_SLC__1SDV_20190715T160437_20190715T160504_028130_032D5B_58D6_Orb_Stack_coh_deb_data"};

    Metadata metadata(
        coh_1_data + "/S1A_IW_SLC__1SDV_20190715T160437_20190715T160504_028130_032D5B_58D6_Orb_Stack_coh_deb.dim",
        coh_1_data + "/latitude.img", coh_1_data + "/longitude.img");
    Dataset<double> input(coh_1_tif);

    auto egm_96 = std::make_shared<EarthGravitationalModel96>();
    egm_96->HostToDevice();

    std::vector<std::string> files{"./goods/srtm_41_01.tif", "./goods/srtm_42_01.tif"};
    auto srtm_3_model = std::make_unique<Srtm3ElevationModel>(files, egm_96);
    srtm_3_model->LoadTiles();
    srtm_3_model->TransferToDevice();

    const auto* d_srtm_3_tiles = srtm_3_model->GetBuffers();
    const size_t srtm_3_tiles_length{2};

    const std::string output_path{"/tmp/tc_saaremaa.tif"};

    {
        const int tile_side_length{1000};
        TerrainCorrection tc(input.GetGdalDataset(), metadata.GetMetadata(), metadata.GetLatTiePointGrid(),
                             metadata.GetLonTiePointGrid(), d_srtm_3_tiles, srtm_3_tiles_length,
                             srtm_3_model->GetProperties(), alus::dem::Type::SRTM3, srtm_3_model->GetPropertiesValue(),
                             selected_band);
        tc.ExecuteTerrainCorrection(output_path, tile_side_length, tile_side_length);
        const auto output = tc.GetOutputDataset();
    }

    ASSERT_THAT(boost::filesystem::exists(output_path), IsTrue());

    const std::string expected_hash{"3b92490fa52744b9"};
    ASSERT_THAT(alus::utils::test::HashFromBand(output_path), ::testing::Eq(expected_hash));

    CompareGeocoding(
        "./goods/terrain_correction/"
        "S1A_IW_SLC__1SDV_20190715T160437_20190715T160504_028130_032D5B_58D6_Orb_Stack_coh_deb_data/"
        "S1A_IW_SLC__1SDV_20190715T160437_20190715T160504_028130_032D5B_58D6_Orb_Stack_coh_deb_TC.tif",
        output_path);

    CHECK_CUDA_ERR(cudaGetLastError());
    srtm_3_model->ReleaseFromDevice();
    egm_96->DeviceFree();
    cudaDeviceReset();  // for cuda-memcheck --leak-check full
}

TEST_F(TerrainCorrectionIntegrationTest, SaaremaaAverageSceneHeight) {
    const int selected_band{1};
    const bool use_avg_scene_height{true};
    std::string const coh_1_tif{
        "./goods/terrain_correction/"
        "S1A_IW_SLC__1SDV_20190715T160437_20190715T160504_028130_032D5B_58D6_Orb_Stack_coh_deb_data/"
        "S1A_IW_SLC__1SDV_20190715T160437_20190715T160504_028130_032D5B_58D6_Orb_Stack_coh_deb.tif"};
    std::string const coh_1_data{
        "./goods/terrain_correction/"
        "S1A_IW_SLC__1SDV_20190715T160437_20190715T160504_028130_032D5B_58D6_Orb_Stack_coh_deb_data"};

    Metadata metadata(
        coh_1_data + "/S1A_IW_SLC__1SDV_20190715T160437_20190715T160504_028130_032D5B_58D6_Orb_Stack_coh_deb.dim",
        coh_1_data + "/latitude.img", coh_1_data + "/longitude.img");
    Dataset<double> input(coh_1_tif);

    const std::string output_path{"/tmp/tc_saaremaa_avg.tif"};

    {
        const int tile_side_length{1000};
        TerrainCorrection tc(input.GetGdalDataset(), metadata.GetMetadata(), metadata.GetLatTiePointGrid(),
                             metadata.GetLonTiePointGrid(), nullptr, 0, nullptr, alus::dem::Type::SRTM3, {},
                             selected_band, use_avg_scene_height);
        tc.ExecuteTerrainCorrection(output_path, tile_side_length, tile_side_length);
    }

    ASSERT_THAT(boost::filesystem::exists(output_path), IsTrue());
    const std::string expected_hash{"69ca1a5153bf3f8f"};
    ASSERT_THAT(alus::utils::test::HashFromBand(output_path), ::testing::Eq(expected_hash));

    CompareGeocoding(
        "./goods/terrain_correction/"
        "S1A_IW_SLC__1SDV_20190715T160437_20190715T160504_028130_032D5B_58D6_Orb_Stack_coh_deb_data/"
        "S1A_IW_SLC__1SDV_20190715T160437_20190715T160504_028130_032D5B_58D6_Orb_Stack_coh_deb_TC.tif",
        output_path);

    CHECK_CUDA_ERR(cudaGetLastError());
    cudaDeviceReset();  // for cuda-memcheck --leak-check full
}

TEST_F(TerrainCorrectionIntegrationTest, BeirutExplosion) {
    const int selected_band{1};
    std::string const coh_1_tif{
        "./goods/terrain_correction/Beirut_IW1_6_VH_orb_stack_cor_deb_coh_data/"
        "Beirut_IW1_6_VH_orb_stack_cor_deb_coh.tif"};
    std::string const coh_1_data{
        "./goods/terrain_correction/"
        "Beirut_IW1_6_VH_orb_stack_cor_deb_coh_data"};

    Metadata metadata(coh_1_data + "/Beirut_IW1_6_VH_orb_stack_cor_deb_coh.dim", coh_1_data + "/latitude.img",
                      coh_1_data + "/longitude.img");
    Dataset<double> input(coh_1_tif);

    auto egm_96 = std::make_shared<EarthGravitationalModel96>();
    egm_96->HostToDevice();

    std::vector<std::string> files{"./goods/srtm_43_06.tif", "./goods/srtm_44_06.tif"};
    auto srtm_3_model = std::make_unique<Srtm3ElevationModel>(files, egm_96);
    srtm_3_model->LoadTiles();
    srtm_3_model->TransferToDevice();

    const auto* d_srtm_3_tiles = srtm_3_model->GetBuffers();
    const size_t srtm_3_tiles_length{2};

    const std::string output_path{"/tmp/tc_beirut_test.tif"};

    {
        const int tile_side_length{1000};
        TerrainCorrection tc(input.GetGdalDataset(), metadata.GetMetadata(), metadata.GetLatTiePointGrid(),
                             metadata.GetLonTiePointGrid(), d_srtm_3_tiles, srtm_3_tiles_length,
                             srtm_3_model->GetProperties(), alus::dem::Type::SRTM3, srtm_3_model->GetPropertiesValue(),
                             selected_band);
        tc.ExecuteTerrainCorrection(output_path, tile_side_length, tile_side_length);
    }

    ASSERT_THAT(boost::filesystem::exists(output_path), IsTrue());
    CompareGeocoding(
        "./goods/terrain_correction/Beirut_IW1_6_VH_orb_stack_cor_deb_coh_data/"
        "Beirut_IW1_6_VH_orb_stack_cor_deb_coh_TC.tif",
        output_path);

    const std::string expected_boost_hash{"cbb4839b9aef8c0e"};
    ASSERT_THAT(alus::utils::test::HashFromBand(output_path), ::testing::Eq(expected_boost_hash));

    CHECK_CUDA_ERR(cudaGetLastError());
    srtm_3_model->ReleaseFromDevice();
    egm_96->DeviceFree();

    cudaDeviceReset();  // for cuda-memcheck --leak-check full
}
}  // namespace
