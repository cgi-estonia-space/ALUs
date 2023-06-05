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
#include "topsar_merge.h"

#include <gmock/gmock.h>
#include <boost/filesystem.hpp>

#include <string>
#include <string_view>

#include "custom/gdal_image_reader.h"
#include "custom/gdal_image_writer.h"
#include "snap-core/core/datamodel/pugixml_meta_data_reader.h"
#include "srtm3_elevation_model.h"
#include "terrain_correction.h"
#include "terrain_correction_metadata.h"
#include "test_utils.h"

namespace {

using alus::snapengine::EarthGravitationalModel96;
using alus::snapengine::Product;
using alus::snapengine::PugixmlMetaDataReader;
using alus::snapengine::Srtm3ElevationModel;
using alus::snapengine::custom::GdalImageReader;
using alus::snapengine::custom::GdalImageWriter;
using alus::terraincorrection::Metadata;
using alus::terraincorrection::TerrainCorrection;
using alus::topsarmerge::TopsarMergeOperator;

struct Dimension {
    int width;
    int height;
};

std::string_view FindProductName(std::string_view file_name) {
    const auto file_extension_position = file_name.find(".tif");
    return file_name.substr(0, file_extension_position);
}

class TopsarMergeTest : public ::testing::Test {
public:
    TopsarMergeTest() { ValidateInputs(); }

protected:
    boost::filesystem::path saaremaa_iw1_b3_{
        "./goods/topsar-merge/"
        "S1B_IW_SLC__1SDV_20190709T160353_20190709T160421_017059_020187_0733_iw1_b3.tif"};
    boost::filesystem::path saaremaa_iw2_b3_{
        "./goods/topsar-merge/"
        "S1B_IW_SLC__1SDV_20190709T160353_20190709T160421_017059_020187_0733_iw2_b3.tif"};
    boost::filesystem::path saaremaa_iw2_b4_{
        "./goods/topsar-merge/"
        "S1B_IW_SLC__1SDV_20190709T160353_20190709T160421_017059_020187_0733_iw2_b4.tif"};
    boost::filesystem::path saaremaa_iw3_b5_{
        "./goods/topsar-merge/"
        "S1B_IW_SLC__1SDV_20190709T160353_20190709T160421_017059_020187_0733_iw3_b5.tif"};

    static constexpr std::string_view OUTPUT_FILE_OVERLAP{"/tmp/merge_overlap.tif"};
    static constexpr std::string_view OUTPUT_FILE_NO_OVERLAP{"/tmp/merge_no_overlap.tif"};
    static constexpr std::string_view OUTPUT_FILE_TC{"/tmp/merge_tc.tif"};
    static constexpr std::string_view PRODUCT_TYPE{"SLC"};

    std::shared_ptr<Product> PrepareIW1B3Product() {
        const Dimension input_iw1_b3_size{22889, 1500};
        auto source_product = Product::CreateProduct(FindProductName(saaremaa_iw1_b3_.filename().string()),
                                                     PRODUCT_TYPE, input_iw1_b3_size.width, input_iw1_b3_size.height);
        source_product->SetFileLocation(saaremaa_iw1_b3_);
        source_product->SetMetadataReader(std::make_shared<PugixmlMetaDataReader>(
            FindProductName(saaremaa_iw1_b3_.filename().string()).data() + std::string(".xml")));
        auto source_reader_1 = std::make_shared<GdalImageReader>();
        source_reader_1->Open(source_product->GetFileLocation().generic_path().string(), false, true);
        source_product->SetImageReader(source_reader_1);

        return source_product;
    }

    std::shared_ptr<Product> PrepareIW2B4Product() {
        const Dimension input_iw2_b4_size{26658, 1511};

        auto source_product = Product::CreateProduct(FindProductName(saaremaa_iw2_b4_.filename().string()),
                                                     PRODUCT_TYPE, input_iw2_b4_size.width, input_iw2_b4_size.height);
        source_product->SetFileLocation(saaremaa_iw2_b4_);
        source_product->SetMetadataReader(std::make_shared<PugixmlMetaDataReader>(
            FindProductName(saaremaa_iw2_b4_.filename().string()).data() + std::string(".xml")));
        auto source_reader_2 = std::make_shared<GdalImageReader>();
        source_reader_2->Open(source_product->GetFileLocation().generic_path().string(), false, true);
        source_product->SetImageReader(source_reader_2);
        return source_product;
    }

    std::shared_ptr<Product> PrepareIW3B5Product() {
        const Dimension input_iw3_b5_size{25621, 1516};

        auto source_product = Product::CreateProduct(FindProductName(saaremaa_iw3_b5_.filename().string()),
                                                     PRODUCT_TYPE, input_iw3_b5_size.width, input_iw3_b5_size.height);
        source_product->SetFileLocation(saaremaa_iw3_b5_);
        source_product->SetMetadataReader(std::make_shared<PugixmlMetaDataReader>(
            FindProductName(saaremaa_iw3_b5_.filename().string()).data() + std::string(".xml")));
        auto source_reader_3 = std::make_shared<GdalImageReader>();
        source_reader_3->Open(source_product->GetFileLocation().generic_path().string(), false, true);
        source_product->SetImageReader(source_reader_3);

        return source_product;
    }

    std::shared_ptr<Product> PrepareIW2B3Product() {
        const Dimension input_iw2_b3_size{26658, 1511};

        auto source_product = Product::CreateProduct(FindProductName(saaremaa_iw2_b3_.filename().string()),
                                                     PRODUCT_TYPE, input_iw2_b3_size.width, input_iw2_b3_size.height);
        source_product->SetFileLocation(saaremaa_iw2_b3_);
        source_product->SetMetadataReader(std::make_shared<PugixmlMetaDataReader>(
            FindProductName(saaremaa_iw2_b3_.filename().string()).data() + std::string(".xml")));
        auto source_reader_2 = std::make_shared<GdalImageReader>();
        source_reader_2->Open(source_product->GetFileLocation().generic_path().string(), false, true);
        source_product->SetImageReader(source_reader_2);

        return source_product;
    }

private:
    void ValidateInputs() {
        ASSERT_THAT(boost::filesystem::exists(saaremaa_iw1_b3_), ::testing::IsTrue());
        ASSERT_THAT(boost::filesystem::exists(saaremaa_iw2_b4_), ::testing::IsTrue());
        ASSERT_THAT(boost::filesystem::exists(saaremaa_iw2_b3_), ::testing::IsTrue());
        ASSERT_THAT(boost::filesystem::exists(saaremaa_iw3_b5_), ::testing::IsTrue());
    }
};

TEST_F(TopsarMergeTest, NotOverlappingSubSwathsMerge) {
    {
        auto source_product_1 = PrepareIW1B3Product();
        auto source_product_2 = PrepareIW2B4Product();
        auto source_product_3 = PrepareIW3B5Product();

        // NOLINTNEXTLINE
        TopsarMergeOperator merge_operator({source_product_1, source_product_2, source_product_3}, {"VV"}, 324, 640,
                                           OUTPUT_FILE_OVERLAP);  // NOLINT
        const auto data_writer = std::make_shared<GdalImageWriter>();
        const auto target = merge_operator.GetTargetProduct();
        std::vector<double> geo_transform_out;
        std::string projection_out;
        data_writer->Open(target->GetFileLocation().generic_path().string(), target->GetSceneRasterWidth(),
                          target->GetSceneRasterHeight(), geo_transform_out, projection_out, false);
        target->SetImageWriter(data_writer);
        merge_operator.Compute();
        data_writer->Close();
    }

    ASSERT_THAT(boost::filesystem::exists(OUTPUT_FILE_OVERLAP.data()), ::testing::IsTrue());
    const std::string expected_md5{"cad5da7b60ac469e"};
    ASSERT_THAT(alus::utils::test::HashFromBand(OUTPUT_FILE_OVERLAP), ::testing::Eq(expected_md5));
}

TEST_F(TopsarMergeTest, OverlappingSubSwathsMerge) {
    {
        auto source_product_1 = PrepareIW1B3Product();
        auto source_product_2 = PrepareIW2B3Product();

        TopsarMergeOperator merge_operator({source_product_1, source_product_2}, {"VV"}, 424, 492,  // NOLINT
                                           OUTPUT_FILE_NO_OVERLAP);
        const auto data_writer = std::make_shared<GdalImageWriter>();
        const auto target = merge_operator.GetTargetProduct();
        std::vector<double> geo_transform_out;
        std::string projection_out;
        data_writer->Open(target->GetFileLocation().generic_path().string(), target->GetSceneRasterWidth(),
                          target->GetSceneRasterHeight(), geo_transform_out, projection_out, false);
        target->SetImageWriter(data_writer);
        merge_operator.Compute();
        data_writer->Close();
    }

    ASSERT_THAT(boost::filesystem::exists(OUTPUT_FILE_NO_OVERLAP.data()), ::testing::IsTrue());
    const std::string expected_md5{"f1a42120cbd87b31"};
    ASSERT_THAT(alus::utils::test::HashFromBand(OUTPUT_FILE_NO_OVERLAP), ::testing::Eq(expected_md5));
}

TEST_F(TopsarMergeTest, OverlappingSubSwathsMergeWithTC) {
    {
        auto source_product_1 = PrepareIW1B3Product();
        auto source_product_2 = PrepareIW2B3Product();

        TopsarMergeOperator merge_operator({source_product_1, source_product_2}, {"VV"}, 424, 492,  // NOLINT
                                           OUTPUT_FILE_NO_OVERLAP);
        const auto data_writer = std::make_shared<GdalImageWriter>();
        const auto target = merge_operator.GetTargetProduct();
        std::vector<double> geo_transform_out;
        std::string projection_out;
        data_writer->Open(target->GetFileLocation().generic_path().string(), target->GetSceneRasterWidth(),
                          target->GetSceneRasterHeight(), geo_transform_out, projection_out, true);
        target->SetImageWriter(data_writer);
        merge_operator.Compute();

        // TC
        Metadata tc_metadata(target);

        auto egm_96 = std::make_shared<EarthGravitationalModel96>();
        egm_96->HostToDevice();
        std::vector<std::string> srtm3_files{"./goods/srtm_41_01.tif", "./goods/srtm_42_01.tif"};
        auto srtm_3_model = std::make_unique<Srtm3ElevationModel>(srtm3_files, egm_96);
        srtm_3_model->LoadTiles();
        srtm_3_model->TransferToDevice();
        const auto* d_srtm_3_tiles = srtm_3_model->GetBuffers();
        const size_t srtm_3_tiles_length{2};

        const int selected_band{1};

        TerrainCorrection tc(data_writer->GetDataset(), tc_metadata.GetMetadata(), tc_metadata.GetLatTiePointGrid(),
                             tc_metadata.GetLonTiePointGrid(), d_srtm_3_tiles, srtm_3_tiles_length,
                             srtm_3_model->GetProperties(), alus::dem::Type::SRTM3, srtm_3_model->GetPropertiesValue(),
                             selected_band);
        tc.ExecuteTerrainCorrection(OUTPUT_FILE_TC, 1000, 1000);  // NOLINT
    }

    ASSERT_THAT(boost::filesystem::exists(OUTPUT_FILE_TC.data()), ::testing::IsTrue());
    const auto result_hash = alus::utils::test::HashFromBand(OUTPUT_FILE_TC);
    const std::string expected_md5{"5e924006d1e3adc3"};
    ASSERT_TRUE(result_hash == expected_md5 || result_hash == "a03d8dcbdae2658b") << "actual hash " << result_hash;
}

}  // namespace
