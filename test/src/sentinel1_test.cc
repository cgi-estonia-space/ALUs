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
#include <optional>
#include <stdexcept>
#include <string_view>

#include "../goods/sentinel1_calibrate_data.h"

#include "abstract_metadata.h"
#include "allocators.h"
#include "comparators.h"
#include "gmock/gmock.h"
#include "metadata_element.h"
#include "product_data_utc.h"
#include "sentinel1_utils.h"
#include "tests_common.hpp"

using namespace alus::tests;

namespace {

class Sentinel1UtilsTester {
public:
    double** doppler_rate_2_{nullptr};
    double** doppler_centroid_2_{nullptr};
    double** reference_time_2_{nullptr};
    double** range_depend_doppler_rate_2_{nullptr};

    void Read4Arrays() {
        std::ifstream doppler_rate_reader("./goods/backgeocoding/slaveDopplerRate.txt");
        std::ifstream doppler_centroid_reader("./goods/backgeocoding/slaveDopplerCentroid.txt");
        std::ifstream range_depend_doppler_rate_reader("./goods/backgeocoding/slaveRangeDependDopplerRate.txt");
        std::ifstream reference_time_reader("./goods/backgeocoding/slaveReferenceTime.txt");

        int x, y, i, j;
        doppler_rate_reader >> x >> y;
        doppler_rate_2_ = alus::Allocate2DArray<double>(x, y);

        for (i = 0; i < x; i++) {
            for (j = 0; j < y; j++) {
                doppler_rate_reader >> doppler_rate_2_[i][j];
            }
        }

        doppler_centroid_reader >> x >> y;
        doppler_centroid_2_ = alus::Allocate2DArray<double>(x, y);
        for (i = 0; i < x; i++) {
            for (j = 0; j < y; j++) {
                doppler_centroid_reader >> doppler_centroid_2_[i][j];
            }
        }

        range_depend_doppler_rate_reader >> x >> y;
        range_depend_doppler_rate_2_ = alus::Allocate2DArray<double>(x, y);
        for (i = 0; i < x; i++) {
            for (j = 0; j < y; j++) {
                range_depend_doppler_rate_reader >> range_depend_doppler_rate_2_[i][j];
            }
        }

        reference_time_reader >> x >> y;
        reference_time_2_ = alus::Allocate2DArray<double>(x, y);
        for (i = 0; i < x; i++) {
            for (j = 0; j < y; j++) {
                reference_time_reader >> reference_time_2_[i][j];
            }
        }

        doppler_rate_reader.close();
        doppler_centroid_reader.close();
        range_depend_doppler_rate_reader.close();
        reference_time_reader.close();
    }

    Sentinel1UtilsTester() {}
    ~Sentinel1UtilsTester() {
        if (doppler_rate_2_ != nullptr) {
            delete[] doppler_rate_2_;
        }
        if (doppler_centroid_2_ != nullptr) {
            delete[] doppler_centroid_2_;
        }
        if (range_depend_doppler_rate_2_ != nullptr) {
            delete[] range_depend_doppler_rate_2_;
        }
        if (reference_time_2_ != nullptr) {
            delete[] reference_time_2_;
        }
    }
};

/**
 * Important. Perform test with slave data as master does not have these 5 arrays.
 */
TEST(sentinel1, utils) {
    alus::s1tbx::Sentinel1Utils utils;
    utils.SetPlaceHolderFiles("./goods/backgeocoding/slaveOrbitStateVectors.txt",
                              "./goods/backgeocoding/dcEstimateList.txt", "./goods/backgeocoding/azimuthList.txt",
                              "./goods/backgeocoding/slaveBurstLineTimes.txt",
                              "./goods/backgeocoding/slaveGeoLocation.txt");
    utils.ReadPlaceHolderFiles();
    Sentinel1UtilsTester tester;
    tester.Read4Arrays();

    utils.ComputeDopplerRate();
    utils.ComputeReferenceTime();

    std::cout << "starting comparisons." << '\n';
    ASSERT_TRUE(utils.subswath_[0].doppler_centroid_ != nullptr);
    ASSERT_TRUE(tester.doppler_centroid_2_ != nullptr);

    ASSERT_TRUE(utils.subswath_[0].range_depend_doppler_rate_ != nullptr);
    ASSERT_TRUE(tester.range_depend_doppler_rate_2_ != nullptr);

    ASSERT_TRUE(utils.subswath_[0].reference_time_ != nullptr);
    ASSERT_TRUE(tester.reference_time_2_ != nullptr);

    ASSERT_TRUE(utils.subswath_[0].doppler_rate_ != nullptr);
    ASSERT_TRUE(tester.doppler_rate_2_ != nullptr);

    size_t doppler_count =
        alus::EqualsArrays2Dd(utils.subswath_[0].doppler_rate_, tester.doppler_rate_2_,
                              utils.subswath_[0].num_of_bursts_, utils.subswath_[0].samples_per_burst_);
    EXPECT_EQ(doppler_count, 0) << "Doppler Rates do not match. Mismatches: " << doppler_count << '\n';

    size_t reference_count =
        alus::EqualsArrays2Dd(utils.subswath_[0].reference_time_, tester.reference_time_2_,
                              utils.subswath_[0].num_of_bursts_, utils.subswath_[0].samples_per_burst_);
    EXPECT_EQ(reference_count, 0) << "Reference Times do not match. Mismatches: " << reference_count << '\n';

    size_t range_doppler_count =
        alus::EqualsArrays2Dd(utils.subswath_[0].range_depend_doppler_rate_, tester.range_depend_doppler_rate_2_,
                              utils.subswath_[0].num_of_bursts_, utils.subswath_[0].samples_per_burst_);
    EXPECT_EQ(range_doppler_count, 0) << "Range Dependent Doppler Rates do not match. Mismatches: "
                                      << range_doppler_count << '\n';

    size_t centroids_count =
        alus::EqualsArrays2Dd(utils.subswath_[0].doppler_centroid_, tester.doppler_centroid_2_,
                              utils.subswath_[0].num_of_bursts_, utils.subswath_[0].samples_per_burst_);
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

    const auto dn_element =
        std::make_shared<alus::snapengine::MetadataElement>(alus::snapengine::AbstractMetadata::DN);
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

    const auto dn_element =
        std::make_shared<alus::snapengine::MetadataElement>(alus::snapengine::AbstractMetadata::DN);
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

}  // namespace
