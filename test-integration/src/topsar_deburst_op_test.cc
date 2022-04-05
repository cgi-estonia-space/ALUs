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
#include <cstddef>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <boost/filesystem.hpp>
#include <boost/iostreams/device/mapped_file.hpp>

#include <openssl/md5.h>

#include "gmock/gmock.h"

#include "custom/dimension.h"
#include "custom/gdal_image_reader.h"
#include "custom/gdal_image_writer.h"
#include "snap-core/core/datamodel/pugixml_meta_data_reader.h"
#include "snap-core/core/datamodel/pugixml_meta_data_writer.h"
#include "test_utils.h"
#include "topsar_deburst_op.h"

namespace {

class TOPSARDeburstOpIntegrationTest : public ::testing::Test {
public:
    TOPSARDeburstOpIntegrationTest() = default;

protected:
    // todo: for some reason there were 2 versions in testdata directory (just double check if issues later)
    std::string file_name_out_{
        "S1B_IW_SLC__1SDV_20200730T034254_20200730T034321_022695_02B131_E8DD_split_Orb_Stack_coh_Deb"};
    boost::filesystem::path file_directory_out_{"./goods/topsar_deburst_op/custom-format/" + file_name_out_};

    std::string expected_sha256_tiff_{
        "ea2bfc2a1fe11c27e1efbf061ebc9bdf5e1414982bdb2b7d89a1f6a7b77f7e36"};  // S1B_IW_SLC__1SDV_20200730T034254_20200730T034321_022695_02B131_E8DD_split_Orb_Stack_coh_Deb.tif

    boost::filesystem::path file_location_in_{
        "./goods/topsar_deburst_op/custom-format/"
        "S1B_IW_SLC__1SDV_20200730T034254_20200730T034321_022695_02B131_E8DD_split_Orb_Stack_coh/"
        "S1B_IW_SLC__1SDV_20200730T034254_20200730T034321_022695_02B131_E8DD_split_Orb_Stack_coh.tif"};
    boost::filesystem::path metadata_file_location_in_{
        "./goods/topsar_deburst_op/custom-format/"
        "S1B_IW_SLC__1SDV_20200730T034254_20200730T034321_022695_02B131_E8DD_split_Orb_Stack_coh/"
        "S1B_IW_SLC__1SDV_20200730T034254_20200730T034321_022695_02B131_E8DD_split_Orb_Stack_coh.xml"};
    boost::filesystem::path file_location_out_{file_directory_out_.generic_string() +
                                               boost::filesystem::path::preferred_separator + file_name_out_ + ".tif"};

    void SetUp() override { boost::filesystem::remove_all(file_location_out_.parent_path()); }

    void TearDown() override { boost::filesystem::remove_all(file_location_out_.parent_path()); }
};

TEST_F(TOPSARDeburstOpIntegrationTest, singleSwathBeirut) {
    ASSERT_TRUE(boost::filesystem::exists(file_location_in_));
    ASSERT_FALSE(boost::filesystem::exists(file_location_out_));
    {  // ARTIFICAL SCOPE TO FORCE DESTRUCTORS
        const std::string product_type{"SLC"};
        const alus::snapengine::custom::Dimension product_size{21766, 13500};
        // todo: move to some utility like in snap..
        std::size_t file_extension_pos = file_location_in_.filename().string().find(".tif");
        auto file_name_in = file_location_in_.filename().string().substr(0, file_extension_pos);
        auto source_product = alus::snapengine::Product::CreateProduct(file_name_in, product_type.c_str(),
                                                                       product_size.width, product_size.height);
        source_product->SetFileLocation(file_location_in_);
        source_product->SetMetadataReader(
            std::make_shared<alus::snapengine::PugixmlMetaDataReader>(metadata_file_location_in_.string()));

        // some more workaround code for testing..
        auto data_reader = std::make_shared<alus::snapengine::custom::GdalImageReader>();
        data_reader->Open(source_product->GetFileLocation().generic_path().string(), true, true);
        source_product->SetImageReader(data_reader);

        auto op = alus::s1tbx::TOPSARDeburstOp::CreateTOPSARDeburstOp(source_product);
        boost::filesystem::create_directories(file_directory_out_);
        // following is just for this test purposes, creates directories for compute
        // WRITER SETUP IS TEMPORARY TO MAKE TEST WORK, REAL SOLUTION NEEDS DEDICATED PRODUCT WRITER
        auto data_writer = std::make_shared<alus::snapengine::custom::GdalImageWriter>();
        data_writer->Open(op->GetTargetProduct()->GetFileLocation().generic_path().string(),
                          op->GetTargetProduct()->GetSceneRasterWidth(), op->GetTargetProduct()->GetSceneRasterHeight(),
                          data_reader->GetGeoTransform(), data_reader->GetDataProjection(), false);
        op->GetTargetProduct()->SetImageWriter(data_writer);
        op->Compute();

        data_writer->Close();

        auto out_path = file_directory_out_.generic_string() + boost::filesystem::path::preferred_separator +
                        file_name_out_ + ".tif";
        // alus::GeoTiffWriteFile(data_writer->GetDataset(), std::string_view(out_path));
        ASSERT_TRUE(boost::filesystem::exists(file_directory_out_));
        ASSERT_EQ(expected_sha256_tiff_, alus::utils::test::SHA256FromFile(out_path));
    }
}
}  // namespace