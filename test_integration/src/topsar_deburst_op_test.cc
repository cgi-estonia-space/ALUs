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
#include "snap-core/datamodel/pugixml_meta_data_reader.h"
#include "snap-core/datamodel/pugixml_meta_data_writer.h"
#include "topsar_deburst_op.h"

namespace {

// todo:move to testing utility
std::string Md5FromFile(const std::string& path) {
    unsigned char result[MD5_DIGEST_LENGTH];
    boost::iostreams::mapped_file_source src(path);
    MD5((unsigned char*)src.data(), src.size(), result);
    std::ostringstream sout;
    sout << std::hex << std::setfill('0');
    for (auto c : result) sout << std::setw(2) << static_cast<int>(c);
    return sout.str();
}

class TOPSARDeburstOpIntegrationTest : public ::testing::Test {
public:
    TOPSARDeburstOpIntegrationTest() = default;

protected:
    // todo: for some reason there were 2 versions in testdata directory (just double check if issues later)
    std::string file_name_out_{
        "S1B_IW_SLC__1SDV_20200730T034254_20200730T034321_022695_02B131_E8DD_split_Orb_Stack_coh_Deb"};
    boost::filesystem::path file_directory_out_{"./goods/topsar_deburst_op/custom-format/" + file_name_out_};
    // tie point grids
    boost::filesystem::path output_tie_point_grids_directory_{
        file_directory_out_.generic_path().string() + boost::filesystem::path::preferred_separator + "tie_point_grids"};

    std::vector<std::string> expected_output_tie_point_grid_md5sums_{
        "d84891c035cf46e3a79de6c3c72e94ba",  // elevation_angle.hdr
        "c0461403e40a540dc9b35fb0bed78998",  // elevation_angle.img
        "04e5962499681c5aeae904a7ea3706b3",  // incident_angle.hdr
        "7de2fa8880522c3bbd3dde2406907015",  // incident_angle.img
        "8bba8f83d981fcc8672449e4975777d4",  // latitude.hdr
        "c94010c7e7fc17911372e7f6435814c4",  // latitude.img
        "07c126a13578cd0c62ba3bb9f58f7e7d",  // longitude.hdr
        "4e783e5ddc0ea58fa593759ccdcf6389",  // longitude.img
        "8a6158643b21a649fcc9a41de415069b",  // slant_range_time.hdr
        "377dac9aae306f3d04dbdc807d81778b"   // slant_range_time.img
    };

    boost::filesystem::path output_vector_data_directory_{file_directory_out_.generic_path().string() +
                                                          boost::filesystem::path::preferred_separator + "vector_data"};

    std::vector<std::string> expected_output_vector_data_md5sums_{
        "c21783ffb783a5fd51923988475fcc86",  // ground_control_points.csv
        "3dc66adbc064f14ce0b60df4836a2afe"   // pins.csv
    };

    std::string expected_md5_tiff_{
        "36ca94e52e73835009404bab54cc5cf3"};  // S1B_IW_SLC__1SDV_20200730T034254_20200730T034321_022695_02B131_E8DD_split_Orb_Stack_coh_Deb.tif

    std::string expected_md5_xml_{
        "252e0cef8ab1f61923037ca6c568be32"};  // S1B_IW_SLC__1SDV_20200730T034254_20200730T034321_022695_02B131_E8DD_split_Orb_Stack_coh_Deb.xml

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

TEST_F(TOPSARDeburstOpIntegrationTest, single_swath_beirut) {
    ASSERT_TRUE(boost::filesystem::exists(file_location_in_));
    ASSERT_FALSE(boost::filesystem::exists(file_location_out_));
    {  // ARTIFICAL SCOPE TO FORCE DESTRUCTORS
        const std::string product_type{"SLC"};
        alus::snapengine::custom::Dimension product_size{21766, 13500};
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
        // following is just for this test purposes, creates directories for compute
        op->WriteProductFiles(std::make_shared<alus::snapengine::PugixmlMetaDataWriter>());
        // WRITER SETUP IS TEMPORARY TO MAKE TEST WORK, REAL SOLUTION NEEDS DEDICATED PRODUCT WRITER
        auto data_writer = std::make_shared<alus::snapengine::custom::GdalImageWriter>();
        data_writer->Open(op->GetTargetProduct()->GetFileLocation().generic_path().string(),
                          op->GetTargetProduct()->GetSceneRasterWidth(), op->GetTargetProduct()->GetSceneRasterHeight(),
                          data_reader->GetGeoTransform(), data_reader->GetDataProjection());
        op->GetTargetProduct()->SetImageWriter(data_writer);
        op->Compute();
        ASSERT_TRUE(boost::filesystem::exists(file_directory_out_));
        ASSERT_EQ(expected_md5_tiff_,
                  Md5FromFile(file_directory_out_.generic_string() + boost::filesystem::path::preferred_separator +
                              file_name_out_ + ".tif"));
        ASSERT_EQ(expected_md5_xml_,
                  Md5FromFile(file_directory_out_.generic_string() + boost::filesystem::path::preferred_separator +
                              file_name_out_ + ".xml"));

        // TIE POINT GRID FILES
        size_t count_tie_point_grid_files = 0;
        for (boost::filesystem::directory_entry& file :
             boost::filesystem::directory_iterator(output_tie_point_grids_directory_)) {
            ASSERT_THAT(Md5FromFile(file.path().generic_string()),
                        ::testing::AnyOfArray(expected_output_tie_point_grid_md5sums_));
            if (is_regular_file(file)) {
                count_tie_point_grid_files++;
            }
        }
        // number of files
        ASSERT_THAT(count_tie_point_grid_files, expected_output_tie_point_grid_md5sums_.size());

        // VECTOR FILES
        size_t count_vector_data_files = 0;
        for (boost::filesystem::directory_entry& file :
             boost::filesystem::directory_iterator(output_vector_data_directory_)) {
            ASSERT_THAT(Md5FromFile(file.path().generic_string()),
                        ::testing::AnyOfArray(expected_output_vector_data_md5sums_));
            if (is_regular_file(file)) {
                count_vector_data_files++;
            }
        }
        ASSERT_THAT(count_vector_data_files, expected_output_vector_data_md5sums_.size());
    }
}
}  // namespace