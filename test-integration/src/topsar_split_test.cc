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

#include <boost/geometry.hpp>

#include "gmock/gmock.h"
#include "product.h"
#include "pugixml_meta_data_reader.h"
#include "snap-engine-utilities/engine-utilities/datamodel/metadata/abstract_metadata.h"

#include "apply_orbit_file_op.h"
#include "topsar_split.h"

#include "../../test/include/sentinel1_utils_tester.h"
#include "aoi_burst_extract.h"
#include "apply_orbit_file_op.h"
#include "c16_dataset.h"
#include "snap-core/core/util/system_utils.h"
#include "snap-engine-utilities/engine-utilities/datamodel/metadata/abstract_metadata.h"
#include "target_dataset.h"

namespace {

using testing::Sentinel1UtilsTester;

using ::testing::DoubleEq;
using ::testing::DoubleNear;
using ::testing::Eq;

TEST(TopsarSplit, S1utilsSlave) {
    alus::snapengine::SystemUtils::SetAuxDataPath("goods/apply_orbit_file_op/orbit-files/");
    std::string_view subswath_name = "IW1";
    std::string_view polarisation = "VV";

    std::string_view slave_file =
        "goods/beirut_images/S1A_IW_SLC__1SDV_20200805T034334_20200805T034401_033766_03E9F9_52F6.SAFE";

    alus::topsarsplit::TopsarSplit split_slave(slave_file, subswath_name, polarisation);
    split_slave.Initialize();
    std::shared_ptr<alus::C16Dataset<int16_t>> master_reader = split_slave.GetPixelReader();
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
                                        "./goods/backgeocoding/beirutSlaveGeoLocation.txt", 10, 21);  // NOLINT

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

TEST(TopsarSplit, DatasetOnly) {
    std::string_view slave_file =
        "./goods/beirut_images/S1A_IW_SLC__1SDV_20200805T034334_20200805T034401_033766_03E9F9_52F6.SAFE/"
        "measurement/s1a-iw1-slc-vv-20200805t034334-20200805t034359-033766-03e9f9-004.tiff";
    std::vector<int16_t> i_tile(5);  // NOLINT
    std::vector<int16_t> q_tile(5);  // NOLINT
    std::map<int, int16_t*> bands;
    bands.insert({1, i_tile.data()});
    bands.insert({2, q_tile.data()});

    alus::C16Dataset<int16_t> dataset(slave_file);
    dataset.SetReadingArea({0, 4506, 22102, 4506});   // NOLINT
    dataset.ReadRectangle({1000, 200, 5, 1}, bands);  // NOLINT

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

TEST(TopsarSplit, DatasetOutOfBounds) {
    std::string_view slave_file =
        "./goods/beirut_images/S1A_IW_SLC__1SDV_20200805T034334_20200805T034401_033766_03E9F9_52F6.SAFE/"
        "measurement/s1a-iw1-slc-vv-20200805t034334-20200805t034359-033766-03e9f9-004.tiff";
    std::vector<int16_t> i_tile(5);  // NOLINT
    std::vector<int16_t> q_tile(5);  // NOLINT
    std::map<int, int16_t*> bands;
    bands.insert({1, i_tile.data()});
    bands.insert({2, q_tile.data()});

    alus::C16Dataset<int16_t> dataset(slave_file);
    dataset.SetReadingArea({0, 4506, 22102, 4506});  // NOLINT
    ASSERT_THROW(dataset.ReadRectangle({1000, 200, 50000, 1}, bands), alus::DatasetError);

    ASSERT_THROW(dataset.ReadRectangle({1000, 4500, 20, 800}, bands), alus::DatasetError);
}

TEST(TopsarSplit, S1utilsSlaveCut) {
    alus::snapengine::SystemUtils::SetAuxDataPath("goods/apply_orbit_file_op/orbit-files/");
    std::string_view subswath_name = "IW1";
    std::string_view polarisation = "VV";

    std::string_view slave_file =
        "goods/beirut_images/S1A_IW_SLC__1SDV_20200805T034334_20200805T034401_033766_03E9F9_52F6.SAFE";

    alus::topsarsplit::TopsarSplit split_slave(slave_file, subswath_name, polarisation, 4, 6);  // NOLINT
    split_slave.Initialize();
    std::shared_ptr<alus::C16Dataset<int16_t>> master_reader = split_slave.GetPixelReader();
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
                                        "./goods/backgeocoding/beirutSlavePartialGeoLocation.txt", 10, 21);  // NOLINT

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

TEST(TopsarSplit, S1utilsMaster) {
    alus::snapengine::SystemUtils::SetAuxDataPath("./goods/apply_orbit_file_op/orbit-files/");
    std::string_view subswath_name = "IW1";
    std::string_view polarisation = "VV";
    std::string_view master_file =
        "./goods/beirut_images/S1B_IW_SLC__1SDV_20200730T034254_20200730T034321_022695_02B131_E8DD.SAFE";
    alus::topsarsplit::TopsarSplit split_master(master_file, subswath_name, polarisation);
    split_master.Initialize();
    std::shared_ptr<alus::C16Dataset<int16_t>> master_reader = split_master.GetPixelReader();
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

TEST(TopsarSplit, ExtractsSingleBurstFromAoiWkt) {
    std::string subswath_name = "IW1";
    std::string polarisation = "VV";
    std::string master_file =
        "./goods/beirut_images/S1B_IW_SLC__1SDV_20200730T034254_20200730T034321_022695_02B131_E8DD.SAFE";
    // Only burst 5 selected
    std::string_view aoi_wkt =
        "POLYGON((35.98629227594312 33.78599319266792,35.58951654737135 33.889421777083186,35.55574840025886 "
        "33.83334045091287,35.98629227594312 33.78599319266792))";
    alus::topsarsplit::TopsarSplit split_master(master_file, subswath_name, polarisation, aoi_wkt);
    split_master.Initialize();

    alus::s1tbx::Sentinel1Utils master_utils(split_master.GetTargetProduct());
    const auto& subswath = master_utils.subswath_.at(0);

    EXPECT_THAT(subswath->num_of_bursts_, Eq(1));
    // y offset 6000
    EXPECT_THAT(master_utils.source_image_height_, Eq(1500));
    EXPECT_THAT(master_utils.first_line_utc_, DoubleEq(7516.154935741955));
    // 7516.1549714048961 - Real value from Split by Alus.
    EXPECT_THAT(master_utils.last_line_utc_, DoubleNear(7516.154971404906, 1e-10));
    EXPECT_THAT(subswath->num_of_lines_, Eq(1500));
    auto abs_tgt = alus::snapengine::AbstractMetadata::GetAbstractedMetadata(split_master.GetTargetProduct());
    EXPECT_THAT(alus::snapengine::AbstractMetadata::GetAttributeDouble(
                    abs_tgt, alus::snapengine::AbstractMetadata::FIRST_NEAR_LAT),
                DoubleEq(33.88976728973444));
    EXPECT_THAT(alus::snapengine::AbstractMetadata::GetAttributeDouble(
                    abs_tgt, alus::snapengine::AbstractMetadata::FIRST_NEAR_LONG),
                DoubleEq(35.81126019432898));
    EXPECT_THAT(alus::snapengine::AbstractMetadata::GetAttributeDouble(
                    abs_tgt, alus::snapengine::AbstractMetadata::LAST_FAR_LAT),
                DoubleEq(33.87092932542821));
    EXPECT_THAT(alus::snapengine::AbstractMetadata::GetAttributeDouble(
                    abs_tgt, alus::snapengine::AbstractMetadata::LAST_FAR_LONG),
                DoubleEq(34.82984801448071));
}

TEST(TopsarSplit, ExtractsMultipleBurstsFromAoiWkt) {
    std::string subswath_name = "IW1";
    std::string polarisation = "VV";
    std::string master_file =
        "./goods/beirut_images/S1B_IW_SLC__1SDV_20200730T034254_20200730T034321_022695_02B131_E8DD.SAFE";
    // Four last bursts covered (6-9)
    std::string_view aoi_wkt =
        "POLYGON((34.42016601562499 33.5047590692261,35.35675048828125 33.72433966174759,35.28259277343749 "
        "32.85651801010955,34.42016601562499 33.5047590692261))";
    alus::topsarsplit::TopsarSplit split_master(master_file, subswath_name, polarisation, aoi_wkt);
    split_master.Initialize();

    alus::s1tbx::Sentinel1Utils master_utils(split_master.GetTargetProduct());
    const auto& subswath = master_utils.subswath_.at(0);

    EXPECT_THAT(subswath->num_of_bursts_, Eq(4));
    EXPECT_THAT(master_utils.source_image_height_, Eq(6000));
    EXPECT_THAT(master_utils.first_line_utc_, DoubleEq(7516.154967622118));
    EXPECT_THAT(master_utils.last_line_utc_, DoubleEq(7516.15509904449));
    EXPECT_THAT(subswath->num_of_lines_, Eq(6000));
    auto abs_tgt = alus::snapengine::AbstractMetadata::GetAbstractedMetadata(split_master.GetTargetProduct());
    EXPECT_THAT(alus::snapengine::AbstractMetadata::GetAttributeDouble(
                    abs_tgt, alus::snapengine::AbstractMetadata::FIRST_NEAR_LAT),
                DoubleEq(33.72244304511345));
    EXPECT_THAT(alus::snapengine::AbstractMetadata::GetAttributeDouble(
                    abs_tgt, alus::snapengine::AbstractMetadata::FIRST_NEAR_LONG),
                DoubleEq(35.78204457711804));
    EXPECT_THAT(alus::snapengine::AbstractMetadata::GetAttributeDouble(
                    abs_tgt, alus::snapengine::AbstractMetadata::LAST_FAR_LAT),
                DoubleEq(33.18729964093926));
    EXPECT_THAT(alus::snapengine::AbstractMetadata::GetAttributeDouble(
                    abs_tgt, alus::snapengine::AbstractMetadata::LAST_FAR_LONG),
                DoubleEq(34.67947070642776));
}

TEST(TopsarSplit, CheckSwathBoundaryAoi) {
    std::string polarisation = "VV";
    std::string master_file =
        "./goods/beirut_images/S1B_IW_SLC__1SDV_20200730T034254_20200730T034321_022695_02B131_E8DD.SAFE";

    std::vector<alus::topsarsplit::SwathPolygon> swaths;
    for (std::string_view swath : {"IW1", "IW2", "IW3"}) {
        alus::topsarsplit::TopsarSplit split(master_file, swath, polarisation);
        split.Initialize();

        swaths.push_back(alus::topsarsplit::ExtractSwathPolygon(split.GetTargetProduct()));
    }

    const auto& sw1 = swaths.at(0);
    const auto& sw2 = swaths.at(1);
    const auto& sw3 = swaths.at(2);

    const std::string wkt_sw1 =
        "POLYGON((35.38970947265624 33.945638452963024,35.62591552734374 33.970697997361626,35.61218261718749 "
        "33.76088200086919,35.3594970703125 33.79284377363183,35.38970947265624 33.945638452963024))";
    const std::string wkt_sw2 =
        "POLYGON((34.69482421875 34.04810808490984,34.859619140625 33.993472995119674,34.7991943359375 "
        "33.71063227149209,34.72778320312499 33.408516828002675,34.54650878906249 33.50933936780059,34.58496093749999 "
        "33.77001515278013,34.69482421875 34.04810808490984))";
    const std::string wkt_sw3 =
        "POLYGON((33.35449218749999 34.88593094075314,33.63464355468749 34.818313145609395,33.541259765625 "
        "34.57442951865275,33.23913574218749 34.619647359797185,33.35449218749999 34.88593094075314))";
    const std::string wkt_sw1_sw2_sw3 =
        "POLYGON((33.7884521484375 34.470335121217474,35.4583740234375 34.30714385628805,35.277099609375 "
        "33.96158628979907,33.4588623046875 34.189085831172406,33.7884521484375 34.470335121217474))";
    const std::string wkt_sw1_sw2 =
        "POLYGON((34.62341308593749 33.29839499061643,35.43640136718749 33.18813395605041,35.52978515624999 "
        "33.495597744865705,34.64538574218749 33.51849923765609,34.62341308593749 33.29839499061643))";

    alus::topsarsplit::Aoi aoi_sw1;
    alus::topsarsplit::Aoi aoi_sw2;
    alus::topsarsplit::Aoi aoi_sw3;
    alus::topsarsplit::Aoi aoi_sw1_sw2_sw3;
    alus::topsarsplit::Aoi aoi_sw1_sw2;

    boost::geometry::read_wkt(wkt_sw1, aoi_sw1);
    boost::geometry::read_wkt(wkt_sw2, aoi_sw2);
    boost::geometry::read_wkt(wkt_sw3, aoi_sw3);
    boost::geometry::read_wkt(wkt_sw1_sw2_sw3, aoi_sw1_sw2_sw3);
    boost::geometry::read_wkt(wkt_sw1_sw2, aoi_sw1_sw2);

    ASSERT_TRUE(alus::topsarsplit::IsWithinSwath(aoi_sw1, sw1));
    ASSERT_FALSE(alus::topsarsplit::IsWithinSwath(aoi_sw1, sw2));
    ASSERT_FALSE(alus::topsarsplit::IsWithinSwath(aoi_sw1, sw3));
    ASSERT_TRUE(alus::topsarsplit::IsCovered(aoi_sw1, sw1));
    ASSERT_FALSE(alus::topsarsplit::IsCovered(aoi_sw1, sw3));
    ASSERT_FALSE(alus::topsarsplit::IsCovered(aoi_sw1, sw3));

    ASSERT_FALSE(alus::topsarsplit::IsWithinSwath(aoi_sw2, sw1));
    ASSERT_TRUE(alus::topsarsplit::IsWithinSwath(aoi_sw2, sw2));
    ASSERT_FALSE(alus::topsarsplit::IsWithinSwath(aoi_sw2, sw3));
    ASSERT_FALSE(alus::topsarsplit::IsCovered(aoi_sw2, sw1));
    ASSERT_TRUE(alus::topsarsplit::IsCovered(aoi_sw2, sw2));
    ASSERT_FALSE(alus::topsarsplit::IsCovered(aoi_sw2, sw3));

    ASSERT_FALSE(alus::topsarsplit::IsWithinSwath(aoi_sw3, sw1));
    ASSERT_FALSE(alus::topsarsplit::IsWithinSwath(aoi_sw3, sw2));
    ASSERT_TRUE(alus::topsarsplit::IsWithinSwath(aoi_sw3, sw3));
    ASSERT_FALSE(alus::topsarsplit::IsCovered(aoi_sw3, sw1));
    ASSERT_FALSE(alus::topsarsplit::IsCovered(aoi_sw3, sw2));
    ASSERT_TRUE(alus::topsarsplit::IsCovered(aoi_sw3, sw3));

    ASSERT_FALSE(alus::topsarsplit::IsWithinSwath(aoi_sw3, sw1));
    ASSERT_FALSE(alus::topsarsplit::IsWithinSwath(aoi_sw3, sw2));
    ASSERT_TRUE(alus::topsarsplit::IsWithinSwath(aoi_sw3, sw3));
    ASSERT_FALSE(alus::topsarsplit::IsCovered(aoi_sw3, sw1));
    ASSERT_FALSE(alus::topsarsplit::IsCovered(aoi_sw3, sw2));
    ASSERT_TRUE(alus::topsarsplit::IsCovered(aoi_sw3, sw3));

    ASSERT_FALSE(alus::topsarsplit::IsWithinSwath(aoi_sw1_sw2_sw3, sw1));
    ASSERT_FALSE(alus::topsarsplit::IsWithinSwath(aoi_sw1_sw2_sw3, sw2));
    ASSERT_FALSE(alus::topsarsplit::IsWithinSwath(aoi_sw1_sw2_sw3, sw3));
    ASSERT_TRUE(alus::topsarsplit::IsCovered(aoi_sw1_sw2_sw3, sw1));
    ASSERT_TRUE(alus::topsarsplit::IsCovered(aoi_sw1_sw2_sw3, sw2));
    ASSERT_TRUE(alus::topsarsplit::IsCovered(aoi_sw1_sw2_sw3, sw3));

    ASSERT_FALSE(alus::topsarsplit::IsWithinSwath(aoi_sw1_sw2, sw1));
    ASSERT_FALSE(alus::topsarsplit::IsWithinSwath(aoi_sw1_sw2, sw2));
    ASSERT_FALSE(alus::topsarsplit::IsWithinSwath(aoi_sw1_sw2, sw3));
    ASSERT_TRUE(alus::topsarsplit::IsCovered(aoi_sw1_sw2, sw1));
    ASSERT_TRUE(alus::topsarsplit::IsCovered(aoi_sw1_sw2, sw2));
    ASSERT_FALSE(alus::topsarsplit::IsCovered(aoi_sw1_sw2, sw3));
}

}  // namespace
