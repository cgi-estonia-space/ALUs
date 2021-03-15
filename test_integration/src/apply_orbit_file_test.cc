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
#include "custom/dimension.h"
#include "snap-core/datamodel/pugixml_meta_data_reader.h"
#include "snap-core/datamodel/pugixml_meta_data_writer.h"
#include "snap-engine-utilities/datamodel/metadata/abstract_metadata.h"

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

    void SetUp() override { boost::filesystem::remove_all(file_location_out_.parent_path()); }

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
}  // namespace