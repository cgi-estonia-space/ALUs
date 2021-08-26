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

#include <memory>

#include <gmock/gmock.h>

#include "pugixml_meta_data_reader.h"
#include "topsar_split.h"

#include "../../test/include/sentinel1_utils_tester.h"
#include "apply_orbit_file_op.h"
#include "c16_dataset.h"
#include "snap-core/util/system_utils.h"

namespace {

TEST(DISABLED_topsar_split, subswaths) {
    alus::topsarsplit::TopsarSplit splitter(
        "goods/beirut_images/S1A_IW_SLC__1SDV_20200805T034334_20200805T034401_033766_03E9F9_52F6_thin.SAFE", "IW1",
        "VV");
    splitter.initialize();
}

TEST(topsar_split, s1utils_slave) {
    alus::snapengine::SystemUtils::SetAuxDataPath("goods/apply_orbit_file_op/orbit-files/");
    std::string_view subswath_name = "IW1";
    std::string_view polarisation = "VV";

    std::string_view slave_file =
        "goods/beirut_images/S1A_IW_SLC__1SDV_20200805T034334_20200805T034401_033766_03E9F9_52F6_thin.SAFE";

    alus::topsarsplit::TopsarSplit split_slave(slave_file, subswath_name, polarisation);
    split_slave.initialize();
    std::shared_ptr<alus::C16Dataset<double>> master_reader = split_slave.GetPixelReader();
    alus::s1tbx::ApplyOrbitFileOp orbit_file_slave(split_slave.GetTargetProduct(), true);
    orbit_file_slave.Initialize();

    alus::s1tbx::Sentinel1Utils slave_utils(split_slave.GetTargetProduct());
    slave_utils.ComputeDopplerRate();
    slave_utils.ComputeReferenceTime();
    slave_utils.subswath_.at(0)->HostToDevice();
    slave_utils.HostToDevice();

    Sentinel1UtilsTester tester;
    tester.Read4Arrays("./goods/backgeocoding/beirutSlaveDopplerRate.txt",
                       "./goods/backgeocoding/beirutSlaveDopplerCentroid.txt",
                       "./goods/backgeocoding/beirutSlaveRangeDependDopplerRate.txt",
                       "./goods/backgeocoding/beirutSlaveReferenceTime.txt");
    tester.ReadOriginalPlaceHolderFiles("./goods/backgeocoding/beirutSlaveBurstLineTimes.txt",
                                        "./goods/backgeocoding/beirutSlaveGeoLocation.txt", 10, 21);

    alus::s1tbx::SubSwathInfo* subswath = slave_utils.subswath_.at(0).get();
    alus::s1tbx::SubSwathInfo* tester_subswath = tester.subswath_.at(0).get();

    EXPECT_EQ(slave_utils.source_image_width_, 22102);
    EXPECT_EQ(slave_utils.source_image_height_, 13518);
    EXPECT_EQ(slave_utils.near_range_on_left_, 1);
    EXPECT_EQ(slave_utils.srgr_flag_, 0);

    EXPECT_DOUBLE_EQ(slave_utils.first_line_utc_, 7522.155263510405);
    EXPECT_DOUBLE_EQ(slave_utils.last_line_utc_, 7522.155554571458);
    EXPECT_DOUBLE_EQ(slave_utils.line_time_interval_, 2.3791160879629606E-8);
    EXPECT_DOUBLE_EQ(slave_utils.near_edge_slant_range_, 799611.1154763321);
    EXPECT_DOUBLE_EQ(slave_utils.wavelength_, 0.05546576);
    EXPECT_DOUBLE_EQ(slave_utils.range_spacing_, 2.329562);
    EXPECT_DOUBLE_EQ(slave_utils.azimuth_spacing_, 13.96968);

    EXPECT_DOUBLE_EQ(subswath->azimuth_time_interval_, 0.002055556299999998);
    EXPECT_EQ(subswath->num_of_bursts_, 9);
    EXPECT_EQ(subswath->lines_per_burst_, 1502);
    EXPECT_EQ(subswath->samples_per_burst_, 22102);
    EXPECT_EQ(subswath->first_valid_pixel_, 387);
    EXPECT_EQ(subswath->last_valid_pixel_, 20775);
    EXPECT_DOUBLE_EQ(subswath->range_pixel_spacing_, 2.329562);
    EXPECT_DOUBLE_EQ(subswath->slr_time_to_first_pixel_, 0.002667215582442478);
    EXPECT_DOUBLE_EQ(subswath->slr_time_to_last_pixel_, 0.002838953224227983);
    EXPECT_DOUBLE_EQ(subswath->first_line_time_, 6.49914214767299E8);
    EXPECT_DOUBLE_EQ(subswath->last_line_time_, 6.49914239914974E8);
    EXPECT_DOUBLE_EQ(subswath->radar_frequency_, 5.40500045433435E9);
    EXPECT_DOUBLE_EQ(subswath->azimuth_steering_rate_, 1.590368784);
    EXPECT_EQ(subswath->num_of_geo_lines_, 10);
    EXPECT_EQ(subswath->num_of_geo_points_per_line_, 21);

    for (int i = 0; i < subswath->num_of_geo_lines_; i++) {
        for (int j = 0; j < subswath->num_of_geo_points_per_line_; j++) {
            EXPECT_DOUBLE_EQ(subswath->azimuth_time_[i][j], tester_subswath->azimuth_time_[i][j]);
            EXPECT_DOUBLE_EQ(subswath->slant_range_time_[i][j], tester_subswath->slant_range_time_[i][j]);
            EXPECT_DOUBLE_EQ(subswath->latitude_[i][j], tester_subswath->latitude_[i][j]);
            EXPECT_DOUBLE_EQ(subswath->longitude_[i][j], tester_subswath->longitude_[i][j]);
            EXPECT_DOUBLE_EQ(subswath->incidence_angle_[i][j], tester_subswath->incidence_angle_[i][j]);
        }
    }

    EXPECT_EQ(subswath->burst_first_line_time_.size(), tester_subswath->burst_first_line_time_.size());
    EXPECT_EQ(subswath->burst_last_line_time_.size(), tester_subswath->burst_last_line_time_.size());

    for (size_t i = 0; i < subswath->burst_first_line_time_.size(); i++) {
        EXPECT_DOUBLE_EQ(subswath->burst_first_line_time_.at(i), tester_subswath->burst_first_line_time_.at(i));
    }

    for (size_t i = 0; i < subswath->burst_last_line_time_.size(); i++) {
        EXPECT_DOUBLE_EQ(subswath->burst_last_line_time_.at(i), tester_subswath->burst_last_line_time_.at(i));
    }

    ASSERT_TRUE(slave_utils.subswath_.at(0)->doppler_centroid_ != nullptr);
    ASSERT_TRUE(tester.doppler_centroid_2_ != nullptr);

    ASSERT_TRUE(slave_utils.subswath_.at(0)->range_depend_doppler_rate_ != nullptr);
    ASSERT_TRUE(tester.range_depend_doppler_rate_2_ != nullptr);

    ASSERT_TRUE(slave_utils.subswath_.at(0)->reference_time_ != nullptr);
    ASSERT_TRUE(tester.reference_time_2_ != nullptr);

    ASSERT_TRUE(slave_utils.subswath_.at(0)->doppler_rate_ != nullptr);
    ASSERT_TRUE(tester.doppler_rate_2_ != nullptr);

    size_t doppler_count = alus::EqualsArrays2Dd(slave_utils.subswath_.at(0)->doppler_rate_, tester.doppler_rate_2_,
                                                 slave_utils.subswath_.at(0)->num_of_bursts_,
                                                 slave_utils.subswath_.at(0)->samples_per_burst_);
    EXPECT_EQ(doppler_count, 0) << "Doppler Rates do not match. Mismatches: " << doppler_count << std::endl;

    size_t reference_count = alus::EqualsArrays2Dd(
        slave_utils.subswath_.at(0)->reference_time_, tester.reference_time_2_,
        slave_utils.subswath_.at(0)->num_of_bursts_, slave_utils.subswath_.at(0)->samples_per_burst_);
    EXPECT_EQ(reference_count, 0) << "Reference Times do not match. Mismatches: " << reference_count << std::endl;

    size_t range_doppler_count = alus::EqualsArrays2Dd(
        slave_utils.subswath_.at(0)->range_depend_doppler_rate_, tester.range_depend_doppler_rate_2_,
        slave_utils.subswath_.at(0)->num_of_bursts_, slave_utils.subswath_.at(0)->samples_per_burst_);
    EXPECT_EQ(range_doppler_count, 0) << "Range Dependent Doppler Rates do not match. Mismatches: "
                                      << range_doppler_count << std::endl;

    size_t centroids_count = alus::EqualsArrays2Dd(
        slave_utils.subswath_.at(0)->doppler_centroid_, tester.doppler_centroid_2_,
        slave_utils.subswath_.at(0)->num_of_bursts_, slave_utils.subswath_.at(0)->samples_per_burst_);
    EXPECT_EQ(centroids_count, 0) << "Doppler Centroids do not match. Mismatches: " << centroids_count << std::endl;
}

TEST(topsar_split, dataset_only) {
    std::string_view slave_file =
        "./goods/beirut_images/S1A_IW_SLC__1SDV_20200805T034334_20200805T034401_033766_03E9F9_52F6_thin.SAFE/"
        "measurement/s1a-iw1-slc-vv-20200805t034334-20200805t034359-033766-03e9f9-004.tiff";
    std::vector<int16_t> i_tile(5);
    std::vector<int16_t> q_tile(5);
    std::map<int, int16_t*> bands;
    bands.insert({1, i_tile.data()});
    bands.insert({2, q_tile.data()});

    alus::C16Dataset<int16_t> dataset(slave_file);
    dataset.SetReadingArea({0, 4506, 22102, 4506});
    dataset.ReadRectangle({1000, 200, 5, 1}, bands);

    EXPECT_EQ(i_tile.at(0), -78);
    EXPECT_EQ(i_tile.at(1), -98);
    EXPECT_EQ(i_tile.at(2), -38);
    EXPECT_EQ(i_tile.at(3), 19);
    EXPECT_EQ(i_tile.at(4), 26);

    EXPECT_EQ(q_tile.at(0), 28);
    EXPECT_EQ(q_tile.at(1), 216);
    EXPECT_EQ(q_tile.at(2), 198);
    EXPECT_EQ(q_tile.at(3), -5);
    EXPECT_EQ(q_tile.at(4), -51);
}

TEST(topsar_split, dataset_out_of_bounds) {
    std::string_view slave_file =
        "./goods/beirut_images/S1A_IW_SLC__1SDV_20200805T034334_20200805T034401_033766_03E9F9_52F6_thin.SAFE/"
        "measurement/s1a-iw1-slc-vv-20200805t034334-20200805t034359-033766-03e9f9-004.tiff";
    std::vector<int16_t> i_tile(5);
    std::vector<int16_t> q_tile(5);
    std::map<int, int16_t*> bands;
    bands.insert({1, i_tile.data()});
    bands.insert({2, q_tile.data()});

    alus::C16Dataset<int16_t> dataset(slave_file);
    dataset.SetReadingArea({0, 4506, 22102, 4506});
    ASSERT_THROW(dataset.ReadRectangle({1000, 200, 50000, 1}, bands), alus::DatasetError);

    ASSERT_THROW(dataset.ReadRectangle({1000, 4500, 20, 800}, bands), alus::DatasetError);
}

TEST(topsar_split, s1utils_slave_cut) {
    alus::snapengine::SystemUtils::SetAuxDataPath("goods/apply_orbit_file_op/orbit-files/");
    std::string_view subswath_name = "IW1";
    std::string_view polarisation = "VV";

    std::string_view slave_file =
        "goods/beirut_images/S1A_IW_SLC__1SDV_20200805T034334_20200805T034401_033766_03E9F9_52F6_thin.SAFE";

    alus::topsarsplit::TopsarSplit split_slave(slave_file, subswath_name, polarisation, 4, 6);
    split_slave.initialize();
    std::shared_ptr<alus::C16Dataset<double>> master_reader = split_slave.GetPixelReader();
    alus::s1tbx::ApplyOrbitFileOp orbit_file_slave(split_slave.GetTargetProduct(), true);
    orbit_file_slave.Initialize();

    alus::s1tbx::Sentinel1Utils slave_utils(split_slave.GetTargetProduct());
    slave_utils.ComputeDopplerRate();
    slave_utils.ComputeReferenceTime();
    slave_utils.subswath_.at(0)->HostToDevice();
    slave_utils.HostToDevice();

    Sentinel1UtilsTester tester;
    tester.Read4Arrays("./goods/backgeocoding/beirutSlavePartialDopplerRate.txt",
                       "./goods/backgeocoding/beirutSlavePartialDopplerCentroid.txt",
                       "./goods/backgeocoding/beirutSlavePartialRangeDependDopplerRate.txt",
                       "./goods/backgeocoding/beirutSlavePartialReferenceTime.txt");
    tester.ReadOriginalPlaceHolderFiles("./goods/backgeocoding/beirutSlavePartialBurstLineTimes.txt",
                                        "./goods/backgeocoding/beirutSlavePartialGeoLocation.txt", 10, 21);

    alus::s1tbx::SubSwathInfo* subswath = slave_utils.subswath_.at(0).get();
    alus::s1tbx::SubSwathInfo* tester_subswath = tester.subswath_.at(0).get();

    EXPECT_EQ(slave_utils.source_image_width_, 22102);
    EXPECT_EQ(slave_utils.source_image_height_, 4506);
    EXPECT_EQ(slave_utils.near_range_on_left_, 1);
    EXPECT_EQ(slave_utils.srgr_flag_, 0);

    EXPECT_DOUBLE_EQ(slave_utils.first_line_utc_, 7522.155359222245);
    EXPECT_DOUBLE_EQ(slave_utils.last_line_utc_, 7522.155458788252);
    EXPECT_DOUBLE_EQ(slave_utils.line_time_interval_, 2.3791160879629606E-8);
    EXPECT_DOUBLE_EQ(slave_utils.near_edge_slant_range_, 799611.1154763321);
    EXPECT_DOUBLE_EQ(slave_utils.wavelength_, 0.05546576);
    EXPECT_DOUBLE_EQ(slave_utils.range_spacing_, 2.329562);
    EXPECT_DOUBLE_EQ(slave_utils.azimuth_spacing_, 13.96968);

    EXPECT_DOUBLE_EQ(subswath->azimuth_time_interval_, 0.002055556299999998);
    EXPECT_EQ(subswath->num_of_bursts_, 3);
    EXPECT_EQ(subswath->lines_per_burst_, 1502);
    EXPECT_EQ(subswath->samples_per_burst_, 22102);
    EXPECT_EQ(subswath->first_valid_pixel_, 311);
    EXPECT_EQ(subswath->last_valid_pixel_, 20775);
    EXPECT_DOUBLE_EQ(subswath->range_pixel_spacing_, 2.329562);
    EXPECT_DOUBLE_EQ(subswath->slr_time_to_first_pixel_, 0.002667215582442478);
    EXPECT_DOUBLE_EQ(subswath->slr_time_to_last_pixel_, 0.002838953224227983);
    EXPECT_DOUBLE_EQ(subswath->first_line_time_, 6.49914223036802E8);
    EXPECT_DOUBLE_EQ(subswath->last_line_time_, 6.49914231639305E8);
    EXPECT_DOUBLE_EQ(subswath->radar_frequency_, 5.40500045433435E9);
    EXPECT_DOUBLE_EQ(subswath->azimuth_steering_rate_, 1.590368784);
    EXPECT_EQ(subswath->num_of_geo_lines_, 10);
    EXPECT_EQ(subswath->num_of_geo_points_per_line_, 21);

    for (int i = 0; i < subswath->num_of_geo_lines_; i++) {
        for (int j = 0; j < subswath->num_of_geo_points_per_line_; j++) {
            EXPECT_DOUBLE_EQ(subswath->azimuth_time_[i][j], tester_subswath->azimuth_time_[i][j]);
            EXPECT_DOUBLE_EQ(subswath->slant_range_time_[i][j], tester_subswath->slant_range_time_[i][j]);
            EXPECT_DOUBLE_EQ(subswath->latitude_[i][j], tester_subswath->latitude_[i][j]);
            EXPECT_DOUBLE_EQ(subswath->longitude_[i][j], tester_subswath->longitude_[i][j]);
            EXPECT_DOUBLE_EQ(subswath->incidence_angle_[i][j], tester_subswath->incidence_angle_[i][j]);
        }
    }

    EXPECT_EQ(subswath->burst_first_line_time_.size(), tester_subswath->burst_first_line_time_.size());
    EXPECT_EQ(subswath->burst_last_line_time_.size(), tester_subswath->burst_last_line_time_.size());

    for (size_t i = 0; i < subswath->burst_first_line_time_.size(); i++) {
        EXPECT_DOUBLE_EQ(subswath->burst_first_line_time_.at(i), tester_subswath->burst_first_line_time_.at(i));
    }

    for (size_t i = 0; i < subswath->burst_last_line_time_.size(); i++) {
        EXPECT_DOUBLE_EQ(subswath->burst_last_line_time_.at(i), tester_subswath->burst_last_line_time_.at(i));
    }

    ASSERT_TRUE(slave_utils.subswath_.at(0)->doppler_centroid_ != nullptr);
    ASSERT_TRUE(tester.doppler_centroid_2_ != nullptr);

    ASSERT_TRUE(slave_utils.subswath_.at(0)->range_depend_doppler_rate_ != nullptr);
    ASSERT_TRUE(tester.range_depend_doppler_rate_2_ != nullptr);

    ASSERT_TRUE(slave_utils.subswath_.at(0)->reference_time_ != nullptr);
    ASSERT_TRUE(tester.reference_time_2_ != nullptr);

    ASSERT_TRUE(slave_utils.subswath_.at(0)->doppler_rate_ != nullptr);
    ASSERT_TRUE(tester.doppler_rate_2_ != nullptr);

    size_t doppler_count = alus::EqualsArrays2Dd(slave_utils.subswath_.at(0)->doppler_rate_, tester.doppler_rate_2_,
                                                 slave_utils.subswath_.at(0)->num_of_bursts_,
                                                 slave_utils.subswath_.at(0)->samples_per_burst_);
    EXPECT_EQ(doppler_count, 0) << "Doppler Rates do not match. Mismatches: " << doppler_count << std::endl;

    size_t reference_count = alus::EqualsArrays2Dd(
        slave_utils.subswath_.at(0)->reference_time_, tester.reference_time_2_,
        slave_utils.subswath_.at(0)->num_of_bursts_, slave_utils.subswath_.at(0)->samples_per_burst_);
    EXPECT_EQ(reference_count, 0) << "Reference Times do not match. Mismatches: " << reference_count << std::endl;

    size_t range_doppler_count = alus::EqualsArrays2Dd(
        slave_utils.subswath_.at(0)->range_depend_doppler_rate_, tester.range_depend_doppler_rate_2_,
        slave_utils.subswath_.at(0)->num_of_bursts_, slave_utils.subswath_.at(0)->samples_per_burst_);
    EXPECT_EQ(range_doppler_count, 0) << "Range Dependent Doppler Rates do not match. Mismatches: "
                                      << range_doppler_count << std::endl;

    size_t centroids_count = alus::EqualsArrays2Dd(
        slave_utils.subswath_.at(0)->doppler_centroid_, tester.doppler_centroid_2_,
        slave_utils.subswath_.at(0)->num_of_bursts_, slave_utils.subswath_.at(0)->samples_per_burst_);
    EXPECT_EQ(centroids_count, 0) << "Doppler Centroids do not match. Mismatches: " << centroids_count << std::endl;
}

TEST(topsar_split, s1utils_master) {
    alus::snapengine::SystemUtils::SetAuxDataPath("./goods/apply_orbit_file_op/orbit-files/");
    std::string_view subswath_name = "IW1";
    std::string_view polarisation = "VV";
    std::string_view master_file =
        "./goods/beirut_images/S1B_IW_SLC__1SDV_20200730T034254_20200730T034321_022695_02B131_E8DD_thin.SAFE";
    alus::topsarsplit::TopsarSplit split_master(master_file, subswath_name, polarisation);
    split_master.initialize();
    std::shared_ptr<alus::C16Dataset<double>> master_reader = split_master.GetPixelReader();
    alus::s1tbx::ApplyOrbitFileOp orbit_file_master(split_master.GetTargetProduct(), true);
    orbit_file_master.Initialize();

    alus::s1tbx::Sentinel1Utils master_utils(split_master.GetTargetProduct());
    master_utils.ComputeDopplerRate();
    master_utils.ComputeReferenceTime();
    master_utils.subswath_.at(0)->HostToDevice();
    master_utils.HostToDevice();

    alus::s1tbx::SubSwathInfo* subswath = master_utils.subswath_.at(0).get();
    // alus::s1tbx::SubSwathInfo* tester_subswath = tester.subswath_.at(0).get();

    EXPECT_DOUBLE_EQ(subswath->azimuth_time_interval_, 0.002055556299999998);
    EXPECT_EQ(subswath->num_of_bursts_, 9);
    EXPECT_EQ(subswath->lines_per_burst_, 1500);
    EXPECT_EQ(subswath->samples_per_burst_, 21766);
    EXPECT_EQ(subswath->first_valid_pixel_, 332);
    EXPECT_EQ(subswath->last_valid_pixel_, 20721);
    EXPECT_DOUBLE_EQ(subswath->range_pixel_spacing_, 2.329562);
    EXPECT_DOUBLE_EQ(subswath->slr_time_to_first_pixel_, 0.0026676414964982233);
    EXPECT_DOUBLE_EQ(subswath->slr_time_to_last_pixel_, 0.0028367682225948483);
    EXPECT_DOUBLE_EQ(subswath->first_line_time_, 6.4939577541799E8);
    EXPECT_DOUBLE_EQ(subswath->last_line_time_, 6.493958005574441E8);
    EXPECT_DOUBLE_EQ(subswath->radar_frequency_, 5.40500045433435E9);
    EXPECT_DOUBLE_EQ(subswath->azimuth_steering_rate_, 1.590368784);
    EXPECT_EQ(subswath->num_of_geo_lines_, 10);
    EXPECT_EQ(subswath->num_of_geo_points_per_line_, 21);

    EXPECT_DOUBLE_EQ(master_utils.first_line_utc_, 7516.154808078588);
    EXPECT_DOUBLE_EQ(master_utils.last_line_utc_, 7516.155099044491);
    EXPECT_DOUBLE_EQ(master_utils.line_time_interval_, 2.3791160879629606E-8);
    EXPECT_DOUBLE_EQ(master_utils.near_edge_slant_range_, 799738.8012980007);
    EXPECT_DOUBLE_EQ(master_utils.wavelength_, 0.05546576);
    EXPECT_DOUBLE_EQ(master_utils.range_spacing_, 2.329562);
    EXPECT_DOUBLE_EQ(master_utils.azimuth_spacing_, 13.97038);

    EXPECT_EQ(master_utils.source_image_width_, 21766);
    EXPECT_EQ(master_utils.source_image_height_, 13500);
    EXPECT_EQ(master_utils.near_range_on_left_, 1);
    EXPECT_EQ(master_utils.srgr_flag_, 0);
}

}  // namespace
