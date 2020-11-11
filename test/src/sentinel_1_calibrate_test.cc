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
#include <gmock/gmock.h>

#include <cstddef>
#include <memory>
#include <set>
#include <stdexcept>
#include <string_view>

#include "../goods/sentinel1_calibrate_data.h"

#include "calibration_info.h"
#include "calibration_vector.h"
#include "general_constants.h"
#include "abstract_metadata.h"
#include "metadata_element.h"
#include "pugixml_meta_data_reader.h"
#include "sentinel1_calibrate.h"

using namespace alus;
using namespace snapengine;
using namespace sentinel1calibrate;
using namespace goods;
using namespace s1tbx;

namespace {
class Sentinel1CalibrateTest : public ::testing::Test {
private:
    static void AddFirstCalibrationFirstVector(std::shared_ptr<MetadataElement> calibration_vector_list_element) {
        const auto vector_element = std::make_shared<MetadataElement>(AbstractMetadata::CALIBRATION_VECTOR);
        calibration_vector_list_element->AddElement(vector_element);

        vector_element->SetAttributeString(AbstractMetadata::AZIMUTH_TIME,
                                           calibrationdata::FIRST_CALIBRATION_FIRST_VECTOR_AZIMUTH_TIME);
        vector_element->SetAttributeInt(AbstractMetadata::LINE, calibrationdata::FIRST_CALIBRATION_FIRST_VECTOR_LINE);

        const auto pixel_element = std::make_shared<MetadataElement>(AbstractMetadata::PIXEL);
        vector_element->AddElement(pixel_element);
        pixel_element->SetAttributeString(AbstractMetadata::PIXEL,
                                          calibrationdata::FIRST_CALIBRATION_FIRST_VECTOR_PIXEL_STRING);
        pixel_element->SetAttributeInt(AbstractMetadata::COUNT,
                                       calibrationdata::FIRST_CALIBRATION_FIRST_VECTOR_PIXEL_COUNT);

        const auto sigma_element = std::make_shared<MetadataElement>(AbstractMetadata::SIGMA_NOUGHT);
        vector_element->AddElement(sigma_element);
        sigma_element->SetAttributeString(AbstractMetadata::SIGMA_NOUGHT,
                                          calibrationdata::FIRST_CALIBRATION_FIRST_VECTOR_SIGMA_STRING);
        sigma_element->SetAttributeInt(AbstractMetadata::COUNT,
                                       calibrationdata::FIRST_CALIBRATION_FIRST_VECTOR_SIGMA_COUNT);

        const auto beta_element = std::make_shared<MetadataElement>(AbstractMetadata::BETA_NOUGHT);
        vector_element->AddElement(beta_element);
        beta_element->SetAttributeString(AbstractMetadata::BETA_NOUGHT,
                                         calibrationdata::FIRST_CALIBRATION_FIRST_VECTOR_BETA_STRING);
        beta_element->SetAttributeInt(AbstractMetadata::COUNT,
                                      calibrationdata::FIRST_CALIBRATION_FIRST_VECTOR_BETA_COUNT);

        const auto gamma_element = std::make_shared<MetadataElement>(AbstractMetadata::GAMMA);
        vector_element->AddElement(gamma_element);
        gamma_element->SetAttributeString(AbstractMetadata::GAMMA,
                                          calibrationdata::FIRST_CALIBRATION_FIRST_VECTOR_GAMMA_STRING);
        gamma_element->SetAttributeInt(AbstractMetadata::COUNT,
                                       calibrationdata::FIRST_CALIBRATION_FIRST_VECTOR_GAMMA_COUNT);

        const auto dn_element = std::make_shared<MetadataElement>(AbstractMetadata::DN);
        vector_element->AddElement(dn_element);
        dn_element->SetAttributeString(AbstractMetadata::DN,
                                       calibrationdata::FIRST_CALIBRATION_FIRST_VECTOR_DN_STRING);
        dn_element->SetAttributeInt(AbstractMetadata::COUNT, calibrationdata::FIRST_CALIBRATION_FIRST_VECTOR_DN_COUNT);
    }

    static void AddFirstCalibrationSecondVector(std::shared_ptr<MetadataElement> calibration_vector_list_element) {
        const auto vector_element = std::make_shared<MetadataElement>(AbstractMetadata::CALIBRATION_VECTOR);
        calibration_vector_list_element->AddElement(vector_element);

        vector_element->SetAttributeString(AbstractMetadata::AZIMUTH_TIME,
                                           calibrationdata::FIRST_CALIBRATION_SECOND_VECTOR_AZIMUTH_TIME);
        vector_element->SetAttributeInt(AbstractMetadata::LINE, calibrationdata::FIRST_CALIBRATION_SECOND_VECTOR_LINE);

        const auto pixel_element = std::make_shared<MetadataElement>(AbstractMetadata::PIXEL);
        vector_element->AddElement(pixel_element);
        pixel_element->SetAttributeString(AbstractMetadata::PIXEL,
                                          calibrationdata::FIRST_CALIBRATION_SECOND_VECTOR_PIXEL_STRING);
        pixel_element->SetAttributeInt(AbstractMetadata::COUNT,
                                       calibrationdata::FIRST_CALIBRATION_SECOND_VECTOR_PIXEL_COUNT);

        const auto sigma_element = std::make_shared<MetadataElement>(AbstractMetadata::SIGMA_NOUGHT);
        vector_element->AddElement(sigma_element);
        sigma_element->SetAttributeString(AbstractMetadata::SIGMA_NOUGHT,
                                          calibrationdata::FIRST_CALIBRATION_SECOND_VECTOR_SIGMA_STRING);
        sigma_element->SetAttributeInt(AbstractMetadata::COUNT,
                                       calibrationdata::FIRST_CALIBRATION_SECOND_VECTOR_SIGMA_COUNT);

        const auto beta_element = std::make_shared<MetadataElement>(AbstractMetadata::BETA_NOUGHT);
        vector_element->AddElement(beta_element);
        beta_element->SetAttributeString(AbstractMetadata::BETA_NOUGHT,
                                         calibrationdata::FIRST_CALIBRATION_SECOND_VECTOR_BETA_STRING);
        beta_element->SetAttributeInt(AbstractMetadata::COUNT,
                                      calibrationdata::FIRST_CALIBRATION_SECOND_VECTOR_BETA_COUNT);

        const auto gamma_element = std::make_shared<MetadataElement>(AbstractMetadata::GAMMA);
        vector_element->AddElement(gamma_element);
        gamma_element->SetAttributeString(AbstractMetadata::GAMMA,
                                          calibrationdata::FIRST_CALIBRATION_SECOND_VECTOR_GAMMA_STRING);
        gamma_element->SetAttributeInt(AbstractMetadata::COUNT,
                                       calibrationdata::FIRST_CALIBRATION_SECOND_VECTOR_GAMMA_COUNT);

        const auto dn_element = std::make_shared<MetadataElement>(AbstractMetadata::DN);
        vector_element->AddElement(dn_element);
        dn_element->SetAttributeString(AbstractMetadata::DN,
                                       calibrationdata::FIRST_CALIBRATION_SECOND_VECTOR_DN_STRING);
        dn_element->SetAttributeInt(AbstractMetadata::COUNT,
                                    calibrationdata::FIRST_CALIBRATION_SECOND_VECTOR_DN_COUNT);
    }

    static void AddSecondCalibrationFirstVector(std::shared_ptr<MetadataElement> calibration_vector_list_element) {
        const auto vector_element = std::make_shared<MetadataElement>(AbstractMetadata::CALIBRATION_VECTOR);
        calibration_vector_list_element->AddElement(vector_element);

        vector_element->SetAttributeString(AbstractMetadata::AZIMUTH_TIME,
                                           calibrationdata::SECOND_CALIBRATION_FIRST_VECTOR_AZIMUTH_TIME);
        vector_element->SetAttributeInt(AbstractMetadata::LINE, calibrationdata::SECOND_CALIBRATION_FIRST_VECTOR_LINE);

        const auto pixel_element = std::make_shared<MetadataElement>(AbstractMetadata::PIXEL);
        vector_element->AddElement(pixel_element);
        pixel_element->SetAttributeString(AbstractMetadata::PIXEL,
                                          calibrationdata::SECOND_CALIBRATION_FIRST_VECTOR_PIXEL_STRING);
        pixel_element->SetAttributeInt(AbstractMetadata::COUNT,
                                       calibrationdata::SECOND_CALIBRATION_FIRST_VECTOR_PIXEL_COUNT);

        const auto sigma_element = std::make_shared<MetadataElement>(AbstractMetadata::SIGMA_NOUGHT);
        vector_element->AddElement(sigma_element);
        sigma_element->SetAttributeString(AbstractMetadata::SIGMA_NOUGHT,
                                          calibrationdata::SECOND_CALIBRATION_FIRST_VECTOR_SIGMA_STRING);
        sigma_element->SetAttributeInt(AbstractMetadata::COUNT,
                                       calibrationdata::SECOND_CALIBRATION_FIRST_VECTOR_SIGMA_COUNT);

        const auto beta_element = std::make_shared<MetadataElement>(AbstractMetadata::BETA_NOUGHT);
        vector_element->AddElement(beta_element);
        beta_element->SetAttributeString(AbstractMetadata::BETA_NOUGHT,
                                         calibrationdata::SECOND_CALIBRATION_FIRST_VECTOR_BETA_STRING);
        beta_element->SetAttributeInt(AbstractMetadata::COUNT,
                                      calibrationdata::SECOND_CALIBRATION_FIRST_VECTOR_BETA_COUNT);

        const auto gamma_element = std::make_shared<MetadataElement>(AbstractMetadata::GAMMA);
        vector_element->AddElement(gamma_element);
        gamma_element->SetAttributeString(AbstractMetadata::GAMMA,
                                          calibrationdata::SECOND_CALIBRATION_FIRST_VECTOR_GAMMA_STRING);
        gamma_element->SetAttributeInt(AbstractMetadata::COUNT,
                                       calibrationdata::SECOND_CALIBRATION_FIRST_VECTOR_GAMMA_COUNT);

        const auto dn_element = std::make_shared<MetadataElement>(AbstractMetadata::DN);
        vector_element->AddElement(dn_element);
        dn_element->SetAttributeString(AbstractMetadata::DN,
                                       calibrationdata::SECOND_CALIBRATION_FIRST_VECTOR_DN_STRING);
        dn_element->SetAttributeInt(AbstractMetadata::COUNT,
                                    calibrationdata::SECOND_CALIBRATION_FIRST_VECTOR_DN_COUNT);
    }

    static void AddSecondCalibrationSecondVector(std::shared_ptr<MetadataElement> calibration_vector_list_element) {
        const auto vector_element = std::make_shared<MetadataElement>(AbstractMetadata::CALIBRATION_VECTOR);
        calibration_vector_list_element->AddElement(vector_element);

        vector_element->SetAttributeString(AbstractMetadata::AZIMUTH_TIME,
                                           calibrationdata::SECOND_CALIBRATION_SECOND_VECTOR_AZIMUTH_TIME);
        vector_element->SetAttributeInt(AbstractMetadata::LINE,
                                        calibrationdata::SECOND_CALIBRATION_SECOND_VECTOR_LINE);

        const auto pixel_element = std::make_shared<MetadataElement>(AbstractMetadata::PIXEL);
        vector_element->AddElement(pixel_element);
        pixel_element->SetAttributeString(AbstractMetadata::PIXEL,
                                          calibrationdata::SECOND_CALIBRATION_SECOND_VECTOR_PIXEL_STRING);
        pixel_element->SetAttributeInt(AbstractMetadata::COUNT,
                                       calibrationdata::SECOND_CALIBRATION_SECOND_VECTOR_PIXEL_COUNT);

        const auto sigma_element = std::make_shared<MetadataElement>(AbstractMetadata::SIGMA_NOUGHT);
        vector_element->AddElement(sigma_element);
        sigma_element->SetAttributeString(AbstractMetadata::SIGMA_NOUGHT,
                                          calibrationdata::SECOND_CALIBRATION_SECOND_VECTOR_SIGMA_STRING);
        sigma_element->SetAttributeInt(AbstractMetadata::COUNT,
                                       calibrationdata::SECOND_CALIBRATION_SECOND_VECTOR_SIGMA_COUNT);

        const auto beta_element = std::make_shared<MetadataElement>(AbstractMetadata::BETA_NOUGHT);
        vector_element->AddElement(beta_element);
        beta_element->SetAttributeString(AbstractMetadata::BETA_NOUGHT,
                                         calibrationdata::SECOND_CALIBRATION_SECOND_VECTOR_BETA_STRING);
        beta_element->SetAttributeInt(AbstractMetadata::COUNT,
                                      calibrationdata::SECOND_CALIBRATION_SECOND_VECTOR_BETA_COUNT);

        const auto gamma_element = std::make_shared<MetadataElement>(AbstractMetadata::GAMMA);
        vector_element->AddElement(gamma_element);
        gamma_element->SetAttributeString(AbstractMetadata::GAMMA,
                                          calibrationdata::SECOND_CALIBRATION_SECOND_VECTOR_GAMMA_STRING);
        gamma_element->SetAttributeInt(AbstractMetadata::COUNT,
                                       calibrationdata::SECOND_CALIBRATION_SECOND_VECTOR_GAMMA_COUNT);

        const auto dn_element = std::make_shared<MetadataElement>(AbstractMetadata::DN);
        vector_element->AddElement(dn_element);
        dn_element->SetAttributeString(AbstractMetadata::DN,
                                       calibrationdata::SECOND_CALIBRATION_SECOND_VECTOR_DN_STRING);
        dn_element->SetAttributeInt(AbstractMetadata::COUNT,
                                    calibrationdata::SECOND_CALIBRATION_SECOND_VECTOR_DN_COUNT);
    }

    static void AddFirstCalibrationSetElement(std::shared_ptr<MetadataElement> calibration_root) {
        const auto calibration_data_set_item =
            std::make_shared<MetadataElement>(calibrationdata::FIRST_CALIBRATION_SET_ITEM_NAME);
        calibration_root->AddElement(calibration_data_set_item);

        const auto calibration_element = std::make_shared<MetadataElement>(AbstractMetadata::CALIBRATION);
        calibration_data_set_item->AddElement(calibration_element);
        AddFirstAdsHeader(calibration_element);

        const auto calibration_vector_list_element =
            std::make_shared<MetadataElement>(AbstractMetadata::CALIBRATION_VECTOR_LIST);
        calibration_element->AddElement(calibration_vector_list_element);

        calibration_vector_list_element->SetAttributeInt(AbstractMetadata::COUNT,
                                                         calibrationdata::FIRST_CALIBRATION_VECTOR_COUNT);

        AddFirstCalibrationFirstVector(calibration_vector_list_element);
        AddFirstCalibrationSecondVector(calibration_vector_list_element);
    }
    static void AddFirstAdsHeader(std::shared_ptr<MetadataElement> calibration_element) {
        const auto ads_header_element = std::make_shared<MetadataElement>(AbstractMetadata::ADS_HEADER);

        ads_header_element->SetAttributeString(AbstractMetadata::MISSION_ID,
                                               calibrationdata::FIRST_CALIBRATION_MISSION_ID);
        ads_header_element->SetAttributeString(AbstractMetadata::product_type,
                                               calibrationdata::FIRST_CALIBRATION_PRODUCT_TYPE);
        ads_header_element->SetAttributeString(AbstractMetadata::POLARISATION,
                                               calibrationdata::FIRST_CALIBRATION_POLARISATION);
        ads_header_element->SetAttributeString(AbstractMetadata::MODE, calibrationdata::FIRST_CALIBRATION_MODE);
        ads_header_element->SetAttributeString(AbstractMetadata::swath, calibrationdata::FIRST_CALIBRATION_SWATH);
        ads_header_element->SetAttributeString(AbstractMetadata::START_TIME,
                                               calibrationdata::FIRST_CALIBRATION_START_TIME);
        ads_header_element->SetAttributeString(AbstractMetadata::STOP_TIME,
                                               calibrationdata::FIRST_CALIBRATION_STOP_TIME);
        ads_header_element->SetAttributeInt(AbstractMetadata::ABSOLUTE_ORBIT_NUMBER,
                                            calibrationdata::FIRST_CALIBRATION_ABSOLUTE_ORBIT_NUMBER);
        ads_header_element->SetAttributeInt(AbstractMetadata::MISSION_DATA_TAKE_ID,
                                            calibrationdata::FIRST_CALIBRATION_MISSION_DATA_TAKE_ID);
        ads_header_element->SetAttributeInt(AbstractMetadata::IMAGE_NUMBER,
                                            calibrationdata::FIRST_CALIBRATION_IMAGE_NUMBER);

        calibration_element->AddElement(ads_header_element);
    }

    static void AddSecondAdsHeader(std::shared_ptr<MetadataElement> calibration_element) {
        const auto ads_header_element = std::make_shared<MetadataElement>(AbstractMetadata::ADS_HEADER);

        ads_header_element->SetAttributeString(AbstractMetadata::MISSION_ID,
                                               calibrationdata::SECOND_CALIBRATION_MISSION_ID);
        ads_header_element->SetAttributeString(AbstractMetadata::product_type,
                                               calibrationdata::SECOND_CALIBRATION_PRODUCT_TYPE);
        ads_header_element->SetAttributeString(AbstractMetadata::POLARISATION,
                                               calibrationdata::SECOND_CALIBRATION_POLARISATION);
        ads_header_element->SetAttributeString(AbstractMetadata::MODE, calibrationdata::SECOND_CALIBRATION_MODE);
        ads_header_element->SetAttributeString(AbstractMetadata::swath, calibrationdata::SECOND_CALIBRATION_SWATH);
        ads_header_element->SetAttributeString(AbstractMetadata::START_TIME,
                                               calibrationdata::SECOND_CALIBRATION_START_TIME);
        ads_header_element->SetAttributeString(AbstractMetadata::STOP_TIME,
                                               calibrationdata::SECOND_CALIBRATION_STOP_TIME);
        ads_header_element->SetAttributeInt(AbstractMetadata::ABSOLUTE_ORBIT_NUMBER,
                                            calibrationdata::SECOND_CALIBRATION_ABSOLUTE_ORBIT_NUMBER);
        ads_header_element->SetAttributeInt(AbstractMetadata::MISSION_DATA_TAKE_ID,
                                            calibrationdata::SECOND_CALIBRATION_MISSION_DATA_TAKE_ID);
        ads_header_element->SetAttributeInt(AbstractMetadata::IMAGE_NUMBER,
                                            calibrationdata::SECOND_CALIBRATION_IMAGE_NUMBER);

        calibration_element->AddElement(ads_header_element);
    }

    static void AddSecondCalibrationSetElement(std::shared_ptr<MetadataElement> calibration_root) {
        const auto calibration_data_set_item =
            std::make_shared<MetadataElement>(calibrationdata::SECOND_CALIBRATION_SET_ITEM_NAME);
        calibration_root->AddElement(calibration_data_set_item);

        const auto calibration_element = std::make_shared<MetadataElement>(AbstractMetadata::CALIBRATION);
        calibration_data_set_item->AddElement(calibration_element);
        AddSecondAdsHeader(calibration_element);

        const auto calibration_vector_list_element =
            std::make_shared<MetadataElement>(AbstractMetadata::CALIBRATION_VECTOR_LIST);
        calibration_element->AddElement(calibration_vector_list_element);

        calibration_vector_list_element->SetAttributeInt(AbstractMetadata::COUNT,
                                                         calibrationdata::SECOND_CALIBRATION_VECTOR_COUNT);

        AddSecondCalibrationFirstVector(calibration_vector_list_element);
        AddSecondCalibrationSecondVector(calibration_vector_list_element);
    }

    static void AddFirstAnnotationSetElement(std::shared_ptr<MetadataElement> annotation_root) {
        const auto annotation_data_set_item =
            std::make_shared<MetadataElement>(calibrationdata::FIRST_CALIBRATION_ANNOTATION_NAME);
        annotation_root->AddElement(annotation_data_set_item);

        const auto product_element = std::make_shared<MetadataElement>(AbstractMetadata::product);
        annotation_data_set_item->AddElement(product_element);

        const auto image_annotation_element = std::make_shared<MetadataElement>(AbstractMetadata::IMAGE_ANNOTATION);
        product_element->AddElement(image_annotation_element);

        const auto image_information_element = std::make_shared<MetadataElement>(AbstractMetadata::IMAGE_INFORMATION);
        image_annotation_element->AddElement(image_information_element);

        image_information_element->SetAttributeInt(AbstractMetadata::NUMBER_OF_LINES,
                                                   calibrationdata::FIRST_CALIBRATION_NUMBER_OF_LINES);
    }

    static void AddSecondAnnotationSetElement(std::shared_ptr<MetadataElement> annotation_root) {
        const auto annotation_data_set_item =
            std::make_shared<MetadataElement>(calibrationdata::SECOND_CALIBRATION_ANNOTATION_NAME);
        annotation_root->AddElement(annotation_data_set_item);

        const auto product_element = std::make_shared<MetadataElement>(AbstractMetadata::product);
        annotation_data_set_item->AddElement(product_element);

        const auto image_annotation_element = std::make_shared<MetadataElement>(AbstractMetadata::IMAGE_ANNOTATION);
        product_element->AddElement(image_annotation_element);

        const auto image_information_element = std::make_shared<MetadataElement>(AbstractMetadata::IMAGE_INFORMATION);
        image_annotation_element->AddElement(image_information_element);

        image_information_element->SetAttributeInt(AbstractMetadata::NUMBER_OF_LINES,
                                                   calibrationdata::SECOND_CALIBRATION_NUMBER_OF_LINES);
    }

protected:
    std::shared_ptr<MetadataElement> original_product_metadata_;

public:
    Sentinel1CalibrateTest() {
        original_product_metadata_ = std::make_shared<MetadataElement>(AbstractMetadata::ORIGINAL_PRODUCT_METADATA);
        const auto calibration_root_element = std::make_shared<MetadataElement>(AbstractMetadata::CALIBRATION_ROOT);
        original_product_metadata_->AddElement(calibration_root_element);

        AddFirstCalibrationSetElement(calibration_root_element);
        AddSecondCalibrationSetElement(calibration_root_element);

        const auto annotation_root_element = std::make_shared<MetadataElement>(AbstractMetadata::ANNOTATION);
        original_product_metadata_->AddElement(annotation_root_element);

        AddFirstAnnotationSetElement(annotation_root_element);
        AddSecondAnnotationSetElement(annotation_root_element);
    }
};

void CompareCalibrationVectors(const CalibrationVector& actual, const CalibrationVector& expected) {
    ASSERT_THAT(actual.time_mjd, ::testing::DoubleEq(expected.time_mjd));
    ASSERT_THAT(actual.line, ::testing::Eq(expected.line));
    ASSERT_THAT(actual.pixels, ::testing::ContainerEq(expected.pixels));
    ASSERT_THAT(actual.sigma_nought, ::testing::ContainerEq(expected.sigma_nought));
    ASSERT_THAT(actual.beta_nought, ::testing::ContainerEq(expected.beta_nought));
    ASSERT_THAT(actual.gamma, ::testing::ContainerEq(expected.gamma));
    ASSERT_THAT(actual.dn, ::testing::ContainerEq(expected.dn));
    ASSERT_THAT(actual.array_size, ::testing::Eq(expected.array_size));
}

void CompareCalibrationInfo(const CalibrationInfo& actual, const CalibrationInfo& expected) {
    ASSERT_THAT(actual.sub_swath, ::testing::StrEq(expected.sub_swath));
    ASSERT_THAT(actual.polarisation, ::testing::StrEq(expected.polarisation));
    ASSERT_THAT(actual.first_line_time, ::testing::DoubleEq(expected.first_line_time));
    ASSERT_THAT(actual.last_line_time, ::testing::DoubleEq(expected.last_line_time));
    ASSERT_THAT(actual.line_time_interval, ::testing::DoubleEq(expected.line_time_interval));
    ASSERT_THAT(actual.num_of_lines, ::testing::Eq(expected.num_of_lines));
    ASSERT_THAT(actual.count, ::testing::Eq(expected.count));

    ASSERT_THAT(actual.calibration_vectors, ::testing::SizeIs(expected.calibration_vectors.size()));
    for (size_t i = 0; i < actual.calibration_vectors.size(); ++i) {
        CompareCalibrationVectors(actual.calibration_vectors.at(i), expected.calibration_vectors.at(i));
    }
}

TEST_F(Sentinel1CalibrateTest, GetCalibrationInfoList) {
    const std::vector<CalibrationInfo> expected_info_list{calibrationdata::FIRST_CALIBRATION_INFO,
                                                          calibrationdata::SECOND_CALIBRATION_INFO};

    const std::set<std::string_view> selected_polarisations{"VV", "VH"};

    const SelectedCalibrationBands calibration_bands{true, true, true, true};

    const auto result_list =
        GetCalibrationInfoList(original_product_metadata_, selected_polarisations, calibration_bands);

    ASSERT_THAT(result_list, ::testing::SizeIs(expected_info_list.size()));

    for (size_t i = 0; i < result_list.size(); ++i) {
        CompareCalibrationInfo(result_list.at(i), expected_info_list.at(i));
    }
}

TEST_F(Sentinel1CalibrateTest, GetNumOfLines) {
    const auto expeceted_iw2_vv_num_of_lines = calibrationdata::SECOND_CALIBRATION_NUMBER_OF_LINES;
    const auto expeceted_iw1_vh_num_of_lines = calibrationdata::FIRST_CALIBRATION_NUMBER_OF_LINES;
    const auto expected_invalid_number_of_lines = constants::INVALID_INDEX;

    const auto iw1_vh_lines = GetNumOfLines(original_product_metadata_, "vH", "Iw1");
    ASSERT_THAT(iw1_vh_lines, ::testing::Eq(expeceted_iw1_vh_num_of_lines));

    const auto iw2_vv_lines = GetNumOfLines(original_product_metadata_, "vv", "IW2");
    ASSERT_THAT(iw2_vv_lines, ::testing::Eq(expeceted_iw2_vv_num_of_lines));

    const auto iw3_vh_lines = GetNumOfLines(original_product_metadata_, "VH", "IW3");
    ASSERT_THAT(iw3_vh_lines, ::testing::Eq(expected_invalid_number_of_lines));

    const auto iw1_vv_lines = GetNumOfLines(original_product_metadata_, "VV", "IW1");
    ASSERT_THAT(iw1_vv_lines, ::testing::Eq(expected_invalid_number_of_lines));
}

/**
 * Creates and sequentially populates Original Product Metadata and calls GetCalibrationInfoList before each step
 * expecting it to throw an exception.
 */
TEST_F(Sentinel1CalibrateTest, GetExceptions) {
    const auto metadata = std::make_shared<MetadataElement>(AbstractMetadata::ORIGINAL_PRODUCT_METADATA);
    const std::set<std::string_view> selected_polarisations{"VH"};
    const SelectedCalibrationBands calibration_bands{true, false, false, false};
    EXPECT_THROW(GetCalibrationInfoList(metadata, selected_polarisations, calibration_bands), std::runtime_error);

    const auto calibration_root_element = std::make_shared<MetadataElement>(AbstractMetadata::CALIBRATION_ROOT);
    metadata->AddElement(calibration_root_element);
    const auto calibration_data_set_item =
        std::make_shared<MetadataElement>(calibrationdata::FIRST_CALIBRATION_SET_ITEM_NAME);
    calibration_root_element->AddElement(calibration_data_set_item);
    EXPECT_THROW(GetCalibrationInfoList(metadata, selected_polarisations, calibration_bands), std::runtime_error);

    const auto calibration_element = std::make_shared<MetadataElement>(AbstractMetadata::CALIBRATION);
    calibration_data_set_item->AddElement(calibration_element);
    EXPECT_THROW(GetCalibrationInfoList(metadata, selected_polarisations, calibration_bands), std::runtime_error);

    const auto ads_header_element = std::make_shared<MetadataElement>(AbstractMetadata::ADS_HEADER);
    calibration_element->AddElement(ads_header_element);
    EXPECT_THROW(GetCalibrationInfoList(metadata, selected_polarisations, calibration_bands), std::invalid_argument);

    ads_header_element->SetAttributeString(AbstractMetadata::POLARISATION,
                                           calibrationdata::FIRST_CALIBRATION_POLARISATION);
    ads_header_element->SetAttributeString(AbstractMetadata::swath, calibrationdata::FIRST_CALIBRATION_SWATH);
    ads_header_element->SetAttributeString(AbstractMetadata::START_TIME,
                                           calibrationdata::FIRST_CALIBRATION_START_TIME);
    ads_header_element->SetAttributeString(AbstractMetadata::STOP_TIME, calibrationdata::FIRST_CALIBRATION_STOP_TIME);
    EXPECT_THROW(GetCalibrationInfoList(metadata, selected_polarisations, calibration_bands), std::runtime_error);

    const auto annotation_root_element = std::make_shared<MetadataElement>(AbstractMetadata::ANNOTATION);
    metadata->AddElement(annotation_root_element);
    const auto annotation_data_set_item =
        std::make_shared<MetadataElement>(calibrationdata::FIRST_CALIBRATION_ANNOTATION_NAME);
    annotation_root_element->AddElement(annotation_data_set_item);
    EXPECT_THROW(GetCalibrationInfoList(metadata, selected_polarisations, calibration_bands), std::runtime_error);

    const auto product_element = std::make_shared<MetadataElement>(AbstractMetadata::product);
    annotation_data_set_item->AddElement(product_element);
    EXPECT_THROW(GetCalibrationInfoList(metadata, selected_polarisations, calibration_bands), std::runtime_error);

    const auto image_annotation_element = std::make_shared<MetadataElement>(AbstractMetadata::IMAGE_ANNOTATION);
    product_element->AddElement(image_annotation_element);
    EXPECT_THROW(GetCalibrationInfoList(metadata, selected_polarisations, calibration_bands), std::runtime_error);

    const auto image_information_element = std::make_shared<MetadataElement>(AbstractMetadata::IMAGE_INFORMATION);
    image_annotation_element->AddElement(image_information_element);
    EXPECT_THROW(GetCalibrationInfoList(metadata, selected_polarisations, calibration_bands), std::invalid_argument);

    image_information_element->SetAttributeInt(AbstractMetadata::NUMBER_OF_LINES,
                                               calibrationdata::FIRST_CALIBRATION_NUMBER_OF_LINES);
    EXPECT_THROW(GetCalibrationInfoList(metadata, selected_polarisations, calibration_bands), std::runtime_error);

    const auto calibration_vector_list_element =
        std::make_shared<MetadataElement>(AbstractMetadata::CALIBRATION_VECTOR_LIST);
    calibration_element->AddElement(calibration_vector_list_element);
    EXPECT_THROW(GetCalibrationInfoList(metadata, selected_polarisations, calibration_bands), std::invalid_argument);

    calibration_vector_list_element->SetAttributeInt(AbstractMetadata::COUNT,
                                                     calibrationdata::FIRST_CALIBRATION_VECTOR_COUNT);
    EXPECT_THROW(GetCalibrationInfoList(metadata, selected_polarisations, calibration_bands), std::runtime_error);

    const auto vector_element = std::make_shared<MetadataElement>(AbstractMetadata::CALIBRATION_VECTOR);
    calibration_vector_list_element->AddElement(vector_element);
    vector_element->SetAttributeString(AbstractMetadata::AZIMUTH_TIME,
                                       calibrationdata::FIRST_CALIBRATION_FIRST_VECTOR_AZIMUTH_TIME);
    EXPECT_THROW(GetCalibrationInfoList(metadata, selected_polarisations, calibration_bands), std::invalid_argument);

    vector_element->SetAttributeInt(AbstractMetadata::LINE, calibrationdata::FIRST_CALIBRATION_FIRST_VECTOR_LINE);
    EXPECT_THROW(GetCalibrationInfoList(metadata, selected_polarisations, calibration_bands), std::runtime_error);

    const auto pixel_element = std::make_shared<MetadataElement>(AbstractMetadata::PIXEL);
    vector_element->AddElement(pixel_element);
    EXPECT_THROW(GetCalibrationInfoList(metadata, selected_polarisations, calibration_bands), std::invalid_argument);

    pixel_element->SetAttributeString(AbstractMetadata::PIXEL,
                                      calibrationdata::FIRST_CALIBRATION_FIRST_VECTOR_PIXEL_STRING);
    EXPECT_THROW(GetCalibrationInfoList(metadata, selected_polarisations, calibration_bands), std::invalid_argument);

    pixel_element->SetAttributeInt(AbstractMetadata::COUNT,
                                   calibrationdata::FIRST_CALIBRATION_FIRST_VECTOR_PIXEL_COUNT);
    EXPECT_THROW(GetCalibrationInfoList(metadata, selected_polarisations, calibration_bands), std::runtime_error);

    const auto sigma_element = std::make_shared<MetadataElement>(AbstractMetadata::SIGMA_NOUGHT);
    vector_element->AddElement(sigma_element);
    EXPECT_THROW(GetCalibrationInfoList(metadata, selected_polarisations, calibration_bands), std::invalid_argument);
    sigma_element->SetAttributeString(AbstractMetadata::SIGMA_NOUGHT,
                                      calibrationdata::FIRST_CALIBRATION_FIRST_VECTOR_SIGMA_STRING);

    EXPECT_THROW(GetCalibrationInfoList(metadata, selected_polarisations, calibration_bands), std::runtime_error);
    const auto second_vector = original_product_metadata_->GetElement(AbstractMetadata::CALIBRATION_ROOT)
                                   ->GetElement(calibrationdata::FIRST_CALIBRATION_SET_ITEM_NAME)
                                   ->GetElement(AbstractMetadata::CALIBRATION)
                                   ->GetElement(AbstractMetadata::CALIBRATION_VECTOR_LIST)
                                   ->GetElements()
                                   .at(1);
    calibration_vector_list_element->AddElement(second_vector);
    GetCalibrationInfoList(metadata, selected_polarisations, calibration_bands);
}

TEST(Sentinel1CalibrateIntegrationTest, ReadDimTest) {
    const std::vector<CalibrationInfo> expected_info_list{calibrationdata::FIRST_CALIBRATION_INFO,
                                                          calibrationdata::SECOND_CALIBRATION_INFO};

    PugixmlMetaDataReader xml_reader{calibrationdata::TEST_DIM_FILE};
    const auto original_product_metadata = xml_reader.Read(AbstractMetadata::ORIGINAL_PRODUCT_METADATA);

    const std::set<std::string_view> selected_polarisations{"VV", "VH"};

    const SelectedCalibrationBands calibration_bands{true, true, true, true};

    const auto result_list =
        GetCalibrationInfoList(original_product_metadata, selected_polarisations, calibration_bands);

    ASSERT_THAT(result_list, ::testing::SizeIs(expected_info_list.size()));

    for (size_t i = 0; i < result_list.size(); ++i) {
        CompareCalibrationInfo(result_list.at(i), expected_info_list.at(i));
    }
}
}  // namespace
