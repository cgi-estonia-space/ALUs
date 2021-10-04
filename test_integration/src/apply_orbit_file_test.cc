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

#include "apply_orbit_file_op.h"
#include "../goods/apply_orbit_test_data.h"
#include "custom/dimension.h"
#include "sentinel1_product_reader_plug_in.h"
#include "snap-core/datamodel/pugixml_meta_data_reader.h"
#include "snap-core/datamodel/pugixml_meta_data_writer.h"
#include "snap-engine-utilities/datamodel/metadata/abstract_metadata.h"
#include "snap-core/util/alus_utils.h"
#include "snap-core/util/system_utils.h"

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

void VerifyOrbitStateVectorResult(const std::vector<alus::snapengine::OrbitStateVector>& alus_vec,
                                  const std::vector<alus::snapengine::OrbitStateVector>& snap_vec) {
    ASSERT_EQ(alus_vec.size(), snap_vec.size());

    double max_diff[7] = {};
    double max_rel_diff[7] = {};
    for (size_t i = 0; i < alus_vec.size(); i++) {
        const auto& alus = alus_vec[i];
        const auto& snap = snap_vec[i];

        const double diff_mjd = std::abs(alus.time_mjd_ - snap.time_mjd_);
        const double diff_x_pos = std::abs(alus.x_pos_ - snap.x_pos_);
        const double diff_y_pos = std::abs(alus.y_pos_ - snap.y_pos_);
        const double diff_z_pos = std::abs(alus.z_pos_ - snap.z_pos_);
        const double diff_x_vel = std::abs(alus.x_vel_ - snap.x_vel_);
        const double diff_y_vel = std::abs(alus.y_vel_ - snap.y_vel_);
        const double diff_z_vel = std::abs(alus.z_vel_ - snap.z_vel_);

        // store absolute maximum value differences found so far
        max_diff[0] = std::max(diff_mjd, max_diff[0]);
        max_diff[1] = std::max(diff_x_pos, max_diff[1]);
        max_diff[2] = std::max(diff_y_pos, max_diff[2]);
        max_diff[3] = std::max(diff_z_pos, max_diff[3]);
        max_diff[4] = std::max(diff_x_vel, max_diff[4]);
        max_diff[5] = std::max(diff_y_vel, max_diff[5]);
        max_diff[6] = std::max(diff_z_vel, max_diff[6]);

        // store relative maximum value differences found so far
        max_rel_diff[0] = max_diff[0] / snap.time_mjd_;
        max_rel_diff[1] = max_diff[1] / snap.x_pos_;
        max_rel_diff[2] = max_diff[2] / snap.y_pos_;
        max_rel_diff[3] = max_diff[3] / snap.z_pos_;
        max_rel_diff[4] = max_diff[4] / snap.x_vel_;
        max_rel_diff[5] = max_diff[5] / snap.y_vel_;
        max_rel_diff[6] = max_diff[6] / snap.z_vel_;

        // debug printing commented out
        /*
        printf("%.15f , %.15f , %.15f , %.15f , %.15f , %.15f , %.15f\n", alus.time_mjd_, alus.x_pos_,
               alus.y_pos_, alus.z_pos_, alus.x_vel_, alus.y_vel_, alus.z_vel_);
        printf("%.15f , %.15f , %.15f , %.15f , %.15f , %.15f , %.15f\n\n", snap.time_mjd_, snap.x_pos_,
               snap.y_pos_, snap.z_pos_, snap.x_vel_, snap.y_vel_, snap.z_vel_);
        */

    }

    const double MJD_DELTA = 1e-9;
    ASSERT_LE(max_diff[0], MJD_DELTA );

    // compare the biggest position alus vs snap deltas upper bound
    const double POS_DELTA = 0.001;
    ASSERT_LE(max_diff[1], POS_DELTA);
    ASSERT_LE(max_diff[2], POS_DELTA);
    ASSERT_LE(max_diff[3], POS_DELTA);

    // velocity comparison
    const double VEL_DELTA = 0.000001;
    ASSERT_LE(max_diff[4], VEL_DELTA);
    ASSERT_LE(max_diff[5], VEL_DELTA);
    ASSERT_LE(max_diff[6], VEL_DELTA);


    for(double rel_diff : max_rel_diff)
    {
        // the relative error should be less than 1 ppb
        ASSERT_LE(std::abs(rel_diff), 1e-9);
    }

    // debug printing commented out
    /*
    std::cout << "maximum value diffs\n";
    for(double d : max_diff)
    {
        printf("%.15f\n", d);
    }
    std::cout << "Relative diff:\n";
    for(double d : max_rel_diff)
    {
        printf("%.15f\n", d);
    }
    */
}


class ApplyOrbitFileOpIntegrationTest : public ::testing::Test {
public:
    ApplyOrbitFileOpIntegrationTest() = default;

protected:
    std::string file_name_out_{"S1A_IW_SLC__1SDV_20180815T154813_20180815T154840_023259_028747_4563_split_Orb"};
    boost::filesystem::path file_directory_out_{"./goods/apply_orbit_file_op/custom-format/" + file_name_out_};
    // tie point grids
    boost::filesystem::path output_tie_point_grids_directory_{
        file_directory_out_.generic_path().string() + boost::filesystem::path::preferred_separator + "tie_point_grids"};
    std::vector<std::string> expected_output_tie_point_grid_md5sums_{
        "02661f091f30f538bb89ae07e9f7c926",  // elevation_angle.hdr
        "8f4ecc9ec3c39195b5fa0ea2a10046d7",  // elevation_angle.img
        "d139cd2cd472db6c7b87f3a7f45d3484",  // incident_angle.hdr
        "881c78f8183eb204dee1e7d332a46b5c",  // incident_angle.img
        "2b33da8213155d24cdd95fef17112e53",  // latitude.hdr
        "7a853c4107a64049d5d0accef5fb134e",  // latitude.img
        "f4cfcfc41c253f3d69fd23dc3cc732f4",  // longitude.hdr
        "34ab35c27fcb6a797de0e5b65106c08a",  // longitude.img
        "5fe11e3f6b522c32d29f122f8df97d9d",  // slant_range_time.hdr
        "39740a5a09aec7629cfd7bdf27b4861d"   // slant_range_time.img
    };

    boost::filesystem::path output_vector_data_directory_{file_directory_out_.generic_path().string() +
                                                          boost::filesystem::path::preferred_separator + "vector_data"};
    std::vector<std::string> expected_output_vector_data_md5sums_{
        "c21783ffb783a5fd51923988475fcc86",  // ground_control_points.csv
        "3dc66adbc064f14ce0b60df4836a2afe"   // pins.csv
    };

    std::string expected_md5_tiff_{
        "d472b06289859f9d142f057a0c4e1afe"};  // S1A_IW_SLC__1SDV_20180815T154813_20180815T154840_023259_028747_4563_split_Orb.tif
    std::string expected_md5_xml_{
        "0eac9e50c070c21d293d749cd5d5e84c"};  // S1A_IW_SLC__1SDV_20180815T154813_20180815T154840_023259_028747_4563_split_Orb.xml

    boost::filesystem::path file_location_in_{
        "./goods/apply_orbit_file_op/custom-format/"
        "S1A_IW_SLC__1SDV_20180815T154813_20180815T154840_023259_028747_4563_split/"
        "S1A_IW_SLC__1SDV_20180815T154813_20180815T154840_023259_028747_4563_split.tif"};
    boost::filesystem::path file_location_out_{file_directory_out_.generic_string() +
                                               boost::filesystem::path::preferred_separator + file_name_out_ + ".tif"};

    void SetUp() override {
        alus::snapengine::SystemUtils::SetAuxDataPath("./goods/apply_orbit_file_op/orbit-files/");
        boost::filesystem::remove_all(file_location_out_.parent_path());
    }

    void TearDown() override { boost::filesystem::remove_all(file_location_out_.parent_path()); }
};

TEST_F(ApplyOrbitFileOpIntegrationTest, single_burst_data_2018) {
    {  // ARTIFICAL SCOPE TO FORCE DESTRUCTORS
        ASSERT_TRUE(boost::filesystem::exists(file_location_in_));
        ASSERT_FALSE(boost::filesystem::exists(file_location_out_));

        const std::string product_type{"SLC"};
        alus::snapengine::custom::Dimension product_size{21400, 1503};
        // todo: move to some utility like in snap..
        std::size_t file_extension_pos = file_location_in_.filename().string().find(".tif");
        auto file_name_in = file_location_in_.filename().string().substr(0, file_extension_pos);
        auto source_product = alus::snapengine::Product::CreateProduct(file_name_in, product_type.c_str(),
                                                                       product_size.width, product_size.height);
        source_product->SetFileLocation(file_location_in_);
        source_product->SetMetadataReader(std::make_shared<alus::snapengine::PugixmlMetaDataReader>());
        auto operation = alus::s1tbx::ApplyOrbitFileOp(source_product);
        operation.Initialize();
        operation.WriteProductFiles(std::make_shared<alus::snapengine::PugixmlMetaDataWriter>());

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

TEST_F(ApplyOrbitFileOpIntegrationTest, modify_source_only_test) {
    {  // ARTIFICAL SCOPE TO FORCE DESTRUCTORS
        ASSERT_TRUE(boost::filesystem::exists(file_location_in_));
        ASSERT_FALSE(boost::filesystem::exists(file_location_out_));
        alus::snapengine::AlusUtils::SetOrbitFilePath(
            "./goods/apply_orbit_file_op/orbit-files/S1A/2018/08/"
            "S1A_OPER_AUX_POEORB_OPOD_20180904T120748_V20180814T225942_20180816T005942.EOF");

        const std::string product_type{"SLC"};
        alus::snapengine::custom::Dimension product_size{21400, 1503};
        // todo: move to some utility like in snap..
        std::size_t file_extension_pos = file_location_in_.filename().string().find(".tif");
        auto file_name_in = file_location_in_.filename().string().substr(0, file_extension_pos);
        auto source_product = alus::snapengine::Product::CreateProduct(file_name_in, product_type.c_str(),
                                                                       product_size.width, product_size.height);
        source_product->SetFileLocation(file_location_in_);
        source_product->SetMetadataReader(std::make_shared<alus::snapengine::PugixmlMetaDataReader>());
        auto operation = alus::s1tbx::ApplyOrbitFileOp(source_product, true);
        operation.Initialize();

        auto metadata = alus::snapengine::AbstractMetadata::GetAbstractedMetadata(source_product);

        auto orb_vecs = alus::snapengine::AbstractMetadata::GetOrbitStateVectors(metadata);

        ASSERT_NE(metadata->GetAttribute(alus::snapengine::AbstractMetadata::ORBIT_STATE_VECTOR_FILE), nullptr);

        const auto& test_data = OSV_TEST_DATA_S1A_IW_SLC__1SDV_20180815T154813_20180815T154840_023259_028747_4563_split;

        VerifyOrbitStateVectorResult(orb_vecs, test_data);
    }
}

TEST_F(ApplyOrbitFileOpIntegrationTest, modify_safe_test) {
    boost::filesystem::path input_path =
        "./goods/beirut_images/S1A_IW_SLC__1SDV_20200805T034334_20200805T034401_033766_03E9F9_52F6.SAFE";
    alus::snapengine::AlusUtils::SetOrbitFilePath(
        "./goods/apply_orbit_file_op/orbit-files/S1A/2020/08/"
        "S1A_OPER_AUX_POEORB_OPOD_20200825T121215_V20200804T225942_20200806T005942.EOF");

    auto reader_plug_in = std::make_shared<alus::s1tbx::Sentinel1ProductReaderPlugIn>();
    //auto can_read = reader_plug_in->GetDecodeQualification(input_path);
    auto reader = reader_plug_in->CreateReaderInstance();
    auto product = reader->ReadProductNodes(boost::filesystem::canonical("manifest.safe", input_path), nullptr);

    auto operation = alus::s1tbx::ApplyOrbitFileOp(product, true);
    operation.Initialize();

    auto metadata = alus::snapengine::AbstractMetadata::GetAbstractedMetadata(product);
    auto alus_vec = alus::snapengine::AbstractMetadata::GetOrbitStateVectors(metadata);

    VerifyOrbitStateVectorResult(alus_vec, OSV_TEST_DATA_S1A_IW_SLC__1SDV_20200805T034334_20200805T034401_033766_03E9F9_52F6);
}
}  // namespace