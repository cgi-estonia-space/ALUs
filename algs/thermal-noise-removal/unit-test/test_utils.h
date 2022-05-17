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
#pragma once

#include <cstddef>
#include <memory>

#include "gmock/gmock.h"

#include "s1tbx-commons/noise_azimuth_vector.h"
#include "snap-core/core/datamodel/metadata_element.h"
#include "snap-engine-utilities/engine-utilities/datamodel/metadata/abstract_metadata.h"
#include "test_constants.h"
#include "time_maps.h"

namespace alus::tnr::test::utils {

/**
 * Utility function that is used in order to avoid typing same things 100 times.
 *
 * @param name Name of the metadata element.
 * @return Created metadata element.
 */
inline std::shared_ptr<snapengine::MetadataElement> CreateElement(std::string_view name) {
    return std::make_shared<snapengine::MetadataElement>(name);
}

/**
 * Utility function that is used to fill top noise metadata element with azimuthNoiseVector elements.
 *
 * @param parent_noise_element Top noise metadata element.
 * @param image_data Struct with required data, which will be used to fill the metadata.
 */
inline void FillNoiseElementWithAzimuthVectors(const std::shared_ptr<snapengine::MetadataElement>& parent_noise_element,
                                               const constants::ImageData& image_data) {
    const auto image_element = CreateElement(image_data.image_name);
    const auto noise_element = CreateElement(snapengine::AbstractMetadata::NOISE);
    const auto noise_azimuth_vector_list = CreateElement(snapengine::AbstractMetadata::NOISE_AZIMUTH_VECTOR_LIST);

    const auto noise_vector_element = CreateElement(snapengine::AbstractMetadata::NOISE_AZIMUTH_VECTOR);

    const auto line_element = CreateElement(snapengine::AbstractMetadata::LINE);
    line_element->SetAttributeString(snapengine::AbstractMetadata::LINE, image_data.line_attr_value);
    line_element->SetAttributeString(snapengine::AbstractMetadata::COUNT, image_data.line_count);
    noise_vector_element->AddElement(line_element);

    const auto noise_azimuth_lut_element = CreateElement(snapengine::AbstractMetadata::NOISE_AZIMUTH_LUT);
    noise_azimuth_lut_element->SetAttributeString(snapengine::AbstractMetadata::NOISE_AZIMUTH_LUT,
                                                  image_data.noise_azimuth_lut_value);
    noise_azimuth_lut_element->SetAttributeString(snapengine::AbstractMetadata::COUNT,
                                                  image_data.noise_azimuth_lut_count);
    noise_vector_element->AddElement(noise_azimuth_lut_element);

    noise_vector_element->SetAttributeString(snapengine::AbstractMetadata::SWATH, image_data.swath);
    noise_vector_element->SetAttributeString(snapengine::AbstractMetadata::FIRST_AZIMUTH_LINE,
                                             image_data.first_azimuth_line);
    noise_vector_element->SetAttributeString(snapengine::AbstractMetadata::FIRST_RANGE_SAMPLE,
                                             image_data.first_range_sample);
    noise_vector_element->SetAttributeString(snapengine::AbstractMetadata::LAST_AZIMUTH_LINE,
                                             image_data.last_azimuth_line);
    noise_vector_element->SetAttributeString(snapengine::AbstractMetadata::LAST_RANGE_SAMPLE,
                                             image_data.last_range_sample);

    noise_azimuth_vector_list->AddElement(noise_vector_element);
    noise_element->AddElement(noise_azimuth_vector_list);
    image_element->AddElement(noise_element);
    parent_noise_element->AddElement(image_element);
}

/**
 * Fills noiseRangeVectorList metadata element with provided noiseRangeVector metadata.
 *
 * @param noise_range_vector_list noiseRangeVectorList metadata element.
 * @param metadata noiseRangeVector metadata.
 */
inline void FillNoiseRangeVectorList(const std::shared_ptr<snapengine::MetadataElement>& noise_range_vector_list,
                                     const constants::NoiseRangeVectorMetadata& metadata) {
    const auto noise_range_vector = CreateElement(snapengine::AbstractMetadata::NOISE_RANGE_VECTOR);

    const auto pixel_element = CreateElement(snapengine::AbstractMetadata::PIXEL);
    pixel_element->SetAttributeString(snapengine::AbstractMetadata::PIXEL, metadata.pixel);
    pixel_element->SetAttributeString(snapengine::AbstractMetadata::COUNT, metadata.pixel_count);
    noise_range_vector->AddElement(pixel_element);

    const auto noise_range_lut = CreateElement(snapengine::AbstractMetadata::NOISE_RANGE_LUT);
    noise_range_lut->SetAttributeString(snapengine::AbstractMetadata::NOISE_RANGE_LUT, metadata.noise_range_lut);
    noise_range_lut->SetAttributeString(snapengine::AbstractMetadata::COUNT, metadata.noise_range_lut_count);
    noise_range_vector->AddElement(noise_range_lut);

    noise_range_vector->SetAttributeString(snapengine::AbstractMetadata::AZIMUTH_TIME, metadata.azimuth_time);
    noise_range_vector->SetAttributeString(snapengine::AbstractMetadata::LINE, metadata.line);

    noise_range_vector_list->AddElement(noise_range_vector);
}

/**
 * Utility function that is used to fill top noise metadata element with noiseRangeVector elements.
 *
 * @param parent_noise_element Top noise metadata element.
 * @param image_data Struct with required data, which will be used to fill the metadata.
 * @note Should be performed after FillNoiseElementWithAzimuthVectors().
 */
inline void FillNoiseElementWithRangeVectors(const std::shared_ptr<snapengine::MetadataElement>& parent_noise_element,
                                             const constants::ImageData& image_data) {
    const auto noise_element =
        parent_noise_element->GetElement(image_data.image_name)->GetElement(snapengine::AbstractMetadata::NOISE);
    const auto noise_range_vector_list = CreateElement(snapengine::AbstractMetadata::NOISE_RANGE_VECTOR_LIST);

    FillNoiseRangeVectorList(noise_range_vector_list, image_data.noise_range_vector_metadata_1);
    FillNoiseRangeVectorList(noise_range_vector_list, image_data.noise_range_vector_metadata_2);

    noise_element->AddElement(noise_range_vector_list);
}

inline void FillImageInformation(const std::shared_ptr<snapengine::MetadataElement>& original_product_root,
                                 const constants::ImageData& data) {
    std::shared_ptr<snapengine::MetadataElement> annotation =
        original_product_root->GetElement(snapengine::AbstractMetadata::ANNOTATION);
    if (!annotation.get()) {
        annotation = CreateElement(snapengine::AbstractMetadata::ANNOTATION);
        original_product_root->AddElement(annotation);
    }

    const auto image = CreateElement(data.image_name);
    annotation->AddElement(image);

    const auto image_product = CreateElement(snapengine::AbstractMetadata::PRODUCT);
    image->AddElement(image_product);

    const auto image_annotation = CreateElement(snapengine::AbstractMetadata::IMAGE_ANNOTATION);
    image_product->AddElement(image_annotation);

    const auto image_information = CreateElement(snapengine::AbstractMetadata::IMAGE_INFORMATION);
    image_information->SetAttributeString(snapengine::AbstractMetadata::PRODUCT_FIRST_LINE_UTC_TIME,
                                          data.image_information_metadata.product_first_line_utc_time);
    image_information->SetAttributeString(snapengine::AbstractMetadata::AZIMUTH_TIME_INTERVAL,
                                          data.image_information_metadata.azimuth_time_interval);

    image_annotation->AddElement(image_information);
}

/**
 * Creates an original metadata root for testing.
 *
 * @return Original metadata rot.
 */
inline std::shared_ptr<snapengine::MetadataElement> CreateMetadataRoot() {
    auto origin_metadata_root = CreateElement(snapengine::AbstractMetadata::ORIGINAL_PRODUCT_METADATA);
    const auto parent_noise_element = CreateElement(snapengine::AbstractMetadata::NOISE);

    FillNoiseElementWithAzimuthVectors(parent_noise_element, constants::IW1_VV_DATA);
    FillNoiseElementWithAzimuthVectors(parent_noise_element, constants::IW2_VH_DATA);
    FillNoiseElementWithRangeVectors(parent_noise_element, constants::IW1_VV_DATA);
    FillNoiseElementWithRangeVectors(parent_noise_element, constants::IW2_VH_DATA);
    FillImageInformation(origin_metadata_root, constants::IW1_VV_DATA);
    FillImageInformation(origin_metadata_root, constants::IW2_VH_DATA);

    origin_metadata_root->AddElement(parent_noise_element);

    return origin_metadata_root;
}

/**
 * Asserts that computed NoiseAzimuthVector list is the same as the expected_list one.
 *
 * @param expected_list Reference vector.
 * @param computed_list Vector to be controlled.
 */
inline void AssertAzimuthNoiseVectorsAreSame(const std::vector<s1tbx::NoiseAzimuthVector>& expected_list,
                                             const std::vector<s1tbx::NoiseAzimuthVector>& computed_list) {
    ASSERT_THAT(computed_list, ::testing::SizeIs(expected_list.size()));
    for (size_t i = 0; i < expected_list.size(); ++i) {
        const auto expected_vector = expected_list.at(i);
        const auto computed_vector = computed_list.at(i);

        ASSERT_THAT(computed_vector.swath, ::testing::StrEq(expected_vector.swath));
        ASSERT_THAT(computed_vector.first_azimuth_line, ::testing::Eq(expected_vector.first_azimuth_line));
        ASSERT_THAT(computed_vector.last_azimuth_line, ::testing::Eq(expected_vector.last_azimuth_line));
        ASSERT_THAT(computed_vector.first_range_sample, ::testing::Eq(expected_vector.first_range_sample));
        ASSERT_THAT(computed_vector.last_range_sample, ::testing::Eq(expected_vector.last_range_sample));

        // Control, that lines vectors are the same
        ASSERT_THAT(computed_vector.lines, ::testing::SizeIs(expected_vector.lines.size()));
        for (size_t j = 0; j < expected_vector.lines.size(); ++j) {
            ASSERT_THAT(computed_vector.lines.at(j), ::testing::Eq(expected_vector.lines.at(j)));
        }

        // Control, that noise_azimuth_luts are the same
        ASSERT_THAT(computed_vector.noise_azimuth_lut, ::testing::SizeIs(expected_vector.noise_azimuth_lut.size()));
        for (size_t j = 0; j < expected_vector.noise_azimuth_lut.size(); ++j) {
            ASSERT_THAT(computed_vector.noise_azimuth_lut.at(j),
                        ::testing::FloatEq(expected_vector.noise_azimuth_lut.at(j)));
        }
    }
}

/**
 * Asserts that the actual NoiseVector  is the same as the expected one.
 *
 * @param expected Reference NoiseVector.
 * @param actual NoiseVector to be controlled.
 */
inline void AssertNoiseVectorsAreSame(const s1tbx::NoiseVector& expected, const s1tbx::NoiseVector& actual) {
    ASSERT_THAT(actual.time_mjd, ::testing::DoubleEq(expected.time_mjd));
    ASSERT_THAT(actual.line, ::testing::Eq(expected.line));

    // Control, that pixel vectors are the same
    ASSERT_THAT(actual.pixels, ::testing::SizeIs(expected.pixels.size()));
    for (size_t j = 0; j < expected.pixels.size(); ++j) {
        ASSERT_THAT(actual.pixels.at(j), ::testing::Eq(expected.pixels.at(j)));
    }

    // Control, that noise_luts are the same
    ASSERT_THAT(actual.noise_lut, ::testing::SizeIs(expected.noise_lut.size()));
    for (size_t j = 0; j < expected.noise_lut.size(); ++j) {
        ASSERT_THAT(actual.noise_lut.at(j), ::testing::FloatEq(expected.noise_lut.at(j)));
    }
}

/**
 * Asserts that computed NoiseVector list is the same as the expected_list one.
 *
 * @param expected_list Reference vector.
 * @param computed_list Vector to be controlled.
 */
inline void AssertNoiseVectorListsAreSame(const std::vector<s1tbx::NoiseVector>& expected_list,
                                          const std::vector<s1tbx::NoiseVector>& computed_list) {
    ASSERT_THAT(computed_list, ::testing::SizeIs(expected_list.size()));
    for (size_t i = 0; i < expected_list.size(); ++i) {
        AssertNoiseVectorsAreSame(expected_list.at(i), computed_list.at(i));
    }
}

/**
 * Asserts that computed TimeMaps is the same as the expected one.
 * @param expected_map Reference TimeMaps object.
 * @param computed_map TimeMaps object that will be compared to the reference.
 */
inline void AssertTimeMapsAreSame(const TimeMaps& expected_map, const TimeMaps& computed_map) {
    // deltaT map
    ASSERT_THAT(computed_map.delta_t_map, ::testing::SizeIs(expected_map.delta_t_map.size()));
    for (const auto& kv_pair : expected_map.delta_t_map) {
        ASSERT_THAT(computed_map.delta_t_map.count(kv_pair.first), ::testing::Eq(1));
        ASSERT_THAT(computed_map.delta_t_map.at(kv_pair.first), ::testing::DoubleEq(kv_pair.second));
    }

    // T0 map
    ASSERT_THAT(computed_map.t_0_map, ::testing::SizeIs(expected_map.t_0_map.size()));
    for (const auto& kv_pair : expected_map.t_0_map) {
        ASSERT_THAT(computed_map.t_0_map.count(kv_pair.first), ::testing::Eq(1));
        ASSERT_THAT(computed_map.t_0_map.at(kv_pair.first), ::testing::DoubleEq(kv_pair.second));
    }
}
}  // namespace alus::tnr::test::utils