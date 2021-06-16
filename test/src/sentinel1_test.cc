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
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string_view>
#include <vector>

#include "../goods/sentinel1_calibrate_data.h"

#include "abstract_metadata.h"
#include "allocators.h"
#include "comparators.h"
#include "gmock/gmock.h"
#include "metadata_element.h"
#include "product_data_utc.h"
#include "sentinel1_utils.h"
#include "sentinel1_utils_tester.h"
#include "subswath_info.h"
#include "tests_common.hpp"

using namespace alus::tests;

namespace {

/**
 * Important. Perform test with slave data as master does not have these 5 arrays.
 */
TEST(sentinel1, utils) {
    alus::s1tbx::Sentinel1Utils utils("./goods/slave_metadata.dim");
    Sentinel1UtilsTester tester;
    tester.Read4Arrays("./goods/backgeocoding/slaveDopplerRate.txt",
                       "./goods/backgeocoding/slaveDopplerCentroid.txt",
                       "./goods/backgeocoding/slaveRangeDependDopplerRate.txt",
                       "./goods/backgeocoding/slaveReferenceTime.txt");

    utils.ComputeDopplerRate();
    utils.ComputeReferenceTime();

    std::cout << "starting comparisons." << '\n';
    ASSERT_TRUE(utils.subswath_.at(0)->doppler_centroid_ != nullptr);
    ASSERT_TRUE(tester.doppler_centroid_2_ != nullptr);

    ASSERT_TRUE(utils.subswath_.at(0)->range_depend_doppler_rate_ != nullptr);
    ASSERT_TRUE(tester.range_depend_doppler_rate_2_ != nullptr);

    ASSERT_TRUE(utils.subswath_.at(0)->reference_time_ != nullptr);
    ASSERT_TRUE(tester.reference_time_2_ != nullptr);

    ASSERT_TRUE(utils.subswath_.at(0)->doppler_rate_ != nullptr);
    ASSERT_TRUE(tester.doppler_rate_2_ != nullptr);

    size_t doppler_count =
        alus::EqualsArrays2Dd(utils.subswath_.at(0)->doppler_rate_, tester.doppler_rate_2_,
                              utils.subswath_.at(0)->num_of_bursts_, utils.subswath_.at(0)->samples_per_burst_);
    EXPECT_EQ(doppler_count, 0) << "Doppler Rates do not match. Mismatches: " << doppler_count << '\n';

    size_t reference_count =
        alus::EqualsArrays2Dd(utils.subswath_.at(0)->reference_time_, tester.reference_time_2_,
                              utils.subswath_.at(0)->num_of_bursts_, utils.subswath_.at(0)->samples_per_burst_);
    EXPECT_EQ(reference_count, 0) << "Reference Times do not match. Mismatches: " << reference_count << '\n';

    size_t range_doppler_count =
        alus::EqualsArrays2Dd(utils.subswath_.at(0)->range_depend_doppler_rate_, tester.range_depend_doppler_rate_2_,
                              utils.subswath_.at(0)->num_of_bursts_, utils.subswath_.at(0)->samples_per_burst_);
    EXPECT_EQ(range_doppler_count, 0) << "Range Dependent Doppler Rates do not match. Mismatches: "
                                      << range_doppler_count << '\n';

    size_t centroids_count =
        alus::EqualsArrays2Dd(utils.subswath_.at(0)->doppler_centroid_, tester.doppler_centroid_2_,
                              utils.subswath_.at(0)->num_of_bursts_, utils.subswath_.at(0)->samples_per_burst_);
    EXPECT_EQ(centroids_count, 0) << "Doppler Centroids do not match. Mismatches: " << centroids_count << '\n';
}

TEST(Sentinel1Utils, GetTime) {
    const std::string_view input_time_string{"2019-07-09T16:03:53.589741"};
    const std::string_view metadata_element_name{"metadata"};
    const std::string_view test_attribute{"attribute"};
    const double expected_mjd{7129.669370251632};
    const alus::snapengine::Utc expected_utc{7129, 57833, 589741};

    const auto metadata_element = std::make_shared<alus::snapengine::MetadataElement>(metadata_element_name);
    metadata_element->SetAttributeString(test_attribute, input_time_string);

    const auto calculated_utc = alus::s1tbx::Sentinel1Utils::GetTime(metadata_element, test_attribute);
    ASSERT_THAT(calculated_utc->GetMjd(), ::testing::DoubleEq(expected_mjd));
    ASSERT_THAT(calculated_utc->GetDaysFraction(), ::testing::Eq(expected_utc.GetDaysFraction()));
    ASSERT_THAT(calculated_utc->GetSecondsFraction(), ::testing::Eq(expected_utc.GetSecondsFraction()));
    ASSERT_THAT(calculated_utc->GetMicroSecondsFraction(), ::testing::Eq(expected_utc.GetMicroSecondsFraction()));
}

void PrepareFirstCalibrationVector(
    std::shared_ptr<alus::snapengine::MetadataElement> first_calibration_vector_element) {
    first_calibration_vector_element->SetAttributeString(
        alus::snapengine::AbstractMetadata::AZIMUTH_TIME,
        alus::goods::calibrationdata::FIRST_CALIBRATION_SECOND_VECTOR_AZIMUTH_TIME);
    first_calibration_vector_element->SetAttributeInt(
        alus::snapengine::AbstractMetadata::LINE, alus::goods::calibrationdata::FIRST_CALIBRATION_SECOND_VECTOR_LINE);

    const auto pixel_element =
        std::make_shared<alus::snapengine::MetadataElement>(alus::snapengine::AbstractMetadata::PIXEL);
    first_calibration_vector_element->AddElement(pixel_element);
    pixel_element->SetAttributeString(alus::snapengine::AbstractMetadata::PIXEL,
                                      alus::goods::calibrationdata::FIRST_CALIBRATION_SECOND_VECTOR_PIXEL_STRING);
    pixel_element->SetAttributeInt(alus::snapengine::AbstractMetadata::COUNT,
                                   alus::goods::calibrationdata::FIRST_CALIBRATION_SECOND_VECTOR_PIXEL_COUNT);

    const auto sigma_element =
        std::make_shared<alus::snapengine::MetadataElement>(alus::snapengine::AbstractMetadata::SIGMA_NOUGHT);
    first_calibration_vector_element->AddElement(sigma_element);
    sigma_element->SetAttributeString(alus::snapengine::AbstractMetadata::SIGMA_NOUGHT,
                                      alus::goods::calibrationdata::FIRST_CALIBRATION_SECOND_VECTOR_SIGMA_STRING);
    sigma_element->SetAttributeInt(alus::snapengine::AbstractMetadata::COUNT,
                                   alus::goods::calibrationdata::FIRST_CALIBRATION_SECOND_VECTOR_SIGMA_COUNT);

    const auto beta_element =
        std::make_shared<alus::snapengine::MetadataElement>(alus::snapengine::AbstractMetadata::BETA_NOUGHT);
    first_calibration_vector_element->AddElement(beta_element);
    beta_element->SetAttributeString(alus::snapengine::AbstractMetadata::BETA_NOUGHT,
                                     alus::goods::calibrationdata::FIRST_CALIBRATION_SECOND_VECTOR_BETA_STRING);
    beta_element->SetAttributeInt(alus::snapengine::AbstractMetadata::COUNT,
                                  alus::goods::calibrationdata::FIRST_CALIBRATION_SECOND_VECTOR_BETA_COUNT);

    const auto gamma_element =
        std::make_shared<alus::snapengine::MetadataElement>(alus::snapengine::AbstractMetadata::GAMMA);
    first_calibration_vector_element->AddElement(gamma_element);
    gamma_element->SetAttributeString(alus::snapengine::AbstractMetadata::GAMMA,
                                      alus::goods::calibrationdata::FIRST_CALIBRATION_SECOND_VECTOR_GAMMA_STRING);
    gamma_element->SetAttributeInt(alus::snapengine::AbstractMetadata::COUNT,
                                   alus::goods::calibrationdata::FIRST_CALIBRATION_SECOND_VECTOR_GAMMA_COUNT);

    const auto dn_element = std::make_shared<alus::snapengine::MetadataElement>(alus::snapengine::AbstractMetadata::DN);
    first_calibration_vector_element->AddElement(dn_element);
    dn_element->SetAttributeString(alus::snapengine::AbstractMetadata::DN,
                                   alus::goods::calibrationdata::FIRST_CALIBRATION_SECOND_VECTOR_DN_STRING);
    dn_element->SetAttributeInt(alus::snapengine::AbstractMetadata::COUNT,
                                alus::goods::calibrationdata::FIRST_CALIBRATION_SECOND_VECTOR_DN_COUNT);
}

void PrepareSecondCalibrationVector(
    std::shared_ptr<alus::snapengine::MetadataElement> second_calibration_vector_element) {
    second_calibration_vector_element->SetAttributeString(
        alus::snapengine::AbstractMetadata::AZIMUTH_TIME,
        alus::goods::calibrationdata::SECOND_CALIBRATION_FIRST_VECTOR_AZIMUTH_TIME);
    second_calibration_vector_element->SetAttributeInt(
        alus::snapengine::AbstractMetadata::LINE, alus::goods::calibrationdata::SECOND_CALIBRATION_FIRST_VECTOR_LINE);

    const auto pixel_element =
        std::make_shared<alus::snapengine::MetadataElement>(alus::snapengine::AbstractMetadata::PIXEL);
    second_calibration_vector_element->AddElement(pixel_element);
    pixel_element->SetAttributeString(alus::snapengine::AbstractMetadata::PIXEL,
                                      alus::goods::calibrationdata::SECOND_CALIBRATION_FIRST_VECTOR_PIXEL_STRING);
    pixel_element->SetAttributeInt(alus::snapengine::AbstractMetadata::COUNT,
                                   alus::goods::calibrationdata::SECOND_CALIBRATION_FIRST_VECTOR_PIXEL_COUNT);

    const auto sigma_element =
        std::make_shared<alus::snapengine::MetadataElement>(alus::snapengine::AbstractMetadata::SIGMA_NOUGHT);
    second_calibration_vector_element->AddElement(sigma_element);
    sigma_element->SetAttributeString(alus::snapengine::AbstractMetadata::SIGMA_NOUGHT,
                                      alus::goods::calibrationdata::SECOND_CALIBRATION_FIRST_VECTOR_SIGMA_STRING);
    sigma_element->SetAttributeInt(alus::snapengine::AbstractMetadata::COUNT,
                                   alus::goods::calibrationdata::SECOND_CALIBRATION_FIRST_VECTOR_SIGMA_COUNT);

    const auto beta_element =
        std::make_shared<alus::snapengine::MetadataElement>(alus::snapengine::AbstractMetadata::BETA_NOUGHT);
    second_calibration_vector_element->AddElement(beta_element);
    beta_element->SetAttributeString(alus::snapengine::AbstractMetadata::BETA_NOUGHT,
                                     alus::goods::calibrationdata::SECOND_CALIBRATION_FIRST_VECTOR_BETA_STRING);
    beta_element->SetAttributeInt(alus::snapengine::AbstractMetadata::COUNT,
                                  alus::goods::calibrationdata::SECOND_CALIBRATION_FIRST_VECTOR_BETA_COUNT);

    const auto gamma_element =
        std::make_shared<alus::snapengine::MetadataElement>(alus::snapengine::AbstractMetadata::GAMMA);
    second_calibration_vector_element->AddElement(gamma_element);
    gamma_element->SetAttributeString(alus::snapengine::AbstractMetadata::GAMMA,
                                      alus::goods::calibrationdata::SECOND_CALIBRATION_FIRST_VECTOR_GAMMA_STRING);
    gamma_element->SetAttributeInt(alus::snapengine::AbstractMetadata::COUNT,
                                   alus::goods::calibrationdata::SECOND_CALIBRATION_FIRST_VECTOR_GAMMA_COUNT);

    const auto dn_element = std::make_shared<alus::snapengine::MetadataElement>(alus::snapengine::AbstractMetadata::DN);
    second_calibration_vector_element->AddElement(dn_element);
    dn_element->SetAttributeString(alus::snapengine::AbstractMetadata::DN,
                                   alus::goods::calibrationdata::SECOND_CALIBRATION_FIRST_VECTOR_DN_STRING);
    dn_element->SetAttributeInt(alus::snapengine::AbstractMetadata::COUNT,
                                alus::goods::calibrationdata::SECOND_CALIBRATION_FIRST_VECTOR_DN_COUNT);
}

void PrepareCalibrationVectors(std::shared_ptr<alus::snapengine::MetadataElement> calibration_vector_list_element) {
    // Prepare first calibration vector
    const auto first_calibration_vector_element =
        std::make_shared<alus::snapengine::MetadataElement>(alus::snapengine::AbstractMetadata::CALIBRATION_VECTOR);
    calibration_vector_list_element->AddElement(first_calibration_vector_element);
    PrepareFirstCalibrationVector(first_calibration_vector_element);

    // Prepate second calibration vector
    const auto second_calibration_vector_element =
        std::make_shared<alus::snapengine::MetadataElement>(alus::snapengine::AbstractMetadata::CALIBRATION_VECTOR);
    calibration_vector_list_element->AddElement(second_calibration_vector_element);
    PrepareSecondCalibrationVector(second_calibration_vector_element);
}

void CompareCalibrationVectors(const alus::s1tbx::CalibrationVector& actual,
                               const alus::s1tbx::CalibrationVector& expected) {
    ASSERT_THAT(actual.time_mjd, ::testing::DoubleEq(expected.time_mjd));
    ASSERT_THAT(actual.line, ::testing::Eq(expected.line));
    ASSERT_THAT(actual.pixels, ::testing::ContainerEq(expected.pixels));
    ASSERT_THAT(actual.sigma_nought, ::testing::ContainerEq(expected.sigma_nought));
    ASSERT_THAT(actual.beta_nought, ::testing::ContainerEq(expected.beta_nought));
    ASSERT_THAT(actual.gamma, ::testing::ContainerEq(expected.gamma));
    ASSERT_THAT(actual.dn, ::testing::ContainerEq(expected.dn));
    ASSERT_THAT(actual.array_size, ::testing::Eq(expected.array_size));
}

TEST(Sentinel1Utils, GetCalibrationVectors) {
    const auto calibration_vector_list_element =
        std::make_shared<alus::snapengine::MetadataElement>("calibration_vector_list");
    PrepareCalibrationVectors(calibration_vector_list_element);
    const std::vector<alus::s1tbx::CalibrationVector> expected_calibration_vectors{
        alus::goods::calibrationdata::FIRST_CALIBRATION_INFO.calibration_vectors.at(1),
        alus::goods::calibrationdata::SECOND_CALIBRATION_INFO.calibration_vectors.at(0)};

    const auto calculated_vectors =
        alus::s1tbx::Sentinel1Utils::GetCalibrationVectors(calibration_vector_list_element, true, true, true, true);

    ASSERT_THAT(calculated_vectors, ::testing::SizeIs(expected_calibration_vectors.size()));
    for (size_t i = 0; i < calculated_vectors.size(); i++) {
        CompareCalibrationVectors(calculated_vectors.at(i), expected_calibration_vectors.at(i));
    }
}

// Array comparisons omitted on purpose. They are not used on master.
TEST(Sentinel1Utils, MasterTest) {
    alus::s1tbx::Sentinel1Utils master_utils("./goods/master_metadata.dim");

    alus::s1tbx::SubSwathInfo* subswath = master_utils.subswath_.at(0).get();

    EXPECT_DOUBLE_EQ(subswath->azimuth_time_interval_, 0.002055556299999998);
    EXPECT_EQ(subswath->num_of_bursts_, 19);
    EXPECT_EQ(subswath->lines_per_burst_, 1503);
    EXPECT_EQ(subswath->samples_per_burst_, 21401);
    EXPECT_EQ(subswath->first_valid_pixel_, 267);
    EXPECT_EQ(subswath->last_valid_pixel_, 20431);
    EXPECT_DOUBLE_EQ(subswath->range_pixel_spacing_, 2.329562);
    EXPECT_DOUBLE_EQ(subswath->slr_time_to_first_pixel_, 0.002679737321566982);
    EXPECT_DOUBLE_EQ(subswath->slr_time_to_last_pixel_, 0.0028460277850849134);
    // subswath->subswath_name_ = "IW1";  //TODO: come back to this after band names become important
    EXPECT_DOUBLE_EQ(subswath->first_line_time_, 5.49734137546908E8);
    EXPECT_DOUBLE_EQ(subswath->last_line_time_, 5.49734190282205E8);
    EXPECT_DOUBLE_EQ(subswath->radar_frequency_, 5.40500045433435E9);
    EXPECT_DOUBLE_EQ(subswath->azimuth_steering_rate_, 1.590368784);
    EXPECT_EQ(subswath->num_of_geo_lines_, 21);
    EXPECT_EQ(subswath->num_of_geo_points_per_line_, 21);

    EXPECT_DOUBLE_EQ(master_utils.first_line_utc_, 6362.663629015139);
    EXPECT_DOUBLE_EQ(master_utils.last_line_utc_, 6362.664239377373);
    EXPECT_DOUBLE_EQ(master_utils.line_time_interval_, 2.3791160879629606E-8);
    EXPECT_DOUBLE_EQ(master_utils.near_edge_slant_range_, 803365.0384269019);
    EXPECT_DOUBLE_EQ(master_utils.wavelength_, 0.05546576);
    EXPECT_DOUBLE_EQ(master_utils.range_spacing_, 2.329562);
    EXPECT_DOUBLE_EQ(master_utils.azimuth_spacing_, 13.91421);

    EXPECT_EQ(master_utils.source_image_width_, 21401);
    EXPECT_EQ(master_utils.source_image_height_, 28557);
    EXPECT_EQ(master_utils.near_range_on_left_, 1);
    EXPECT_EQ(master_utils.srgr_flag_, 0);
}

TEST(Sentinel1Utils, SlaveTest) {
    alus::s1tbx::Sentinel1Utils slave_utils("./goods/slave_metadata.dim");
    Sentinel1UtilsTester tester;
    tester.ReadOriginalPlaceHolderFiles("./goods/backgeocoding/slaveBurstLineTimes.txt",
                                        "./goods/backgeocoding/slaveGeoLocation.txt", 21, 21);

    alus::s1tbx::SubSwathInfo* subswath = slave_utils.subswath_.at(0).get();
    alus::s1tbx::SubSwathInfo* tester_subswath = tester.subswath_.at(0).get();

    EXPECT_DOUBLE_EQ(subswath->azimuth_time_interval_, 0.002055556299999998);
    EXPECT_EQ(subswath->num_of_bursts_, 19);
    EXPECT_EQ(subswath->lines_per_burst_, 1503);
    EXPECT_EQ(subswath->samples_per_burst_, 21401);
    EXPECT_EQ(subswath->first_valid_pixel_, 267);
    EXPECT_EQ(subswath->last_valid_pixel_, 20431);
    EXPECT_DOUBLE_EQ(subswath->range_pixel_spacing_, 2.329562);
    EXPECT_DOUBLE_EQ(subswath->slr_time_to_first_pixel_, 0.002679737321566982);
    EXPECT_DOUBLE_EQ(subswath->slr_time_to_last_pixel_, 0.0028460277850849134);
    // subswath->subswath_name_ = "IW1";
    EXPECT_DOUBLE_EQ(subswath->first_line_time_, 5.50770938201763E8);
    EXPECT_DOUBLE_EQ(subswath->last_line_time_, 5.50770990939114E8);
    EXPECT_DOUBLE_EQ(subswath->radar_frequency_, 5.40500045433435E9);
    EXPECT_DOUBLE_EQ(subswath->azimuth_steering_rate_, 1.590368784);
    EXPECT_EQ(subswath->num_of_geo_lines_, 21);
    EXPECT_EQ(subswath->num_of_geo_points_per_line_, 21);

    EXPECT_DOUBLE_EQ(slave_utils.first_line_utc_, 6374.66363659448);
    EXPECT_DOUBLE_EQ(slave_utils.last_line_utc_, 6374.6642469804865);
    EXPECT_DOUBLE_EQ(slave_utils.line_time_interval_, 2.3791160879629606E-8);
    EXPECT_DOUBLE_EQ(slave_utils.near_edge_slant_range_, 803365.0384269019);
    EXPECT_DOUBLE_EQ(slave_utils.wavelength_, 0.05546576);
    EXPECT_DOUBLE_EQ(slave_utils.range_spacing_, 2.329562);
    EXPECT_DOUBLE_EQ(slave_utils.azimuth_spacing_, 13.91417);

    EXPECT_EQ(slave_utils.source_image_width_, 21401);
    EXPECT_EQ(slave_utils.source_image_height_, 28557);
    EXPECT_EQ(slave_utils.near_range_on_left_, 1);
    EXPECT_EQ(slave_utils.srgr_flag_, 0);

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
}

}  // namespace
