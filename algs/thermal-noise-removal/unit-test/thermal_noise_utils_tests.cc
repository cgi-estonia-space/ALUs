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

#include "gmock/gmock.h"

#include <memory>
#include <utility>

#include "../include/thermal_noise_utils.h"
#include "snap-core/core/datamodel/metadata_element.h"
#include "test_expected_values.h"
#include "test_utils.h"

namespace test = alus::tnr::test;

namespace {
class ThermalNoiseUtilsTest : public ::testing::Test {
protected:
    std::shared_ptr<alus::snapengine::MetadataElement> original_metadata_root_ = test::utils::CreateMetadataRoot();
};

TEST_F(ThermalNoiseUtilsTest, getAzimuthNoiseVectorListTest) {
    const auto iw1_vv_azimuth_list = original_metadata_root_->GetElement(alus::snapengine::AbstractMetadata::NOISE)
                                         ->GetElement(test::constants::IW1_VV_DATA.image_name)
                                         ->GetElement(alus::snapengine::AbstractMetadata::NOISE)
                                         ->GetElement(alus::snapengine::AbstractMetadata::NOISE_AZIMUTH_VECTOR_LIST);

    const auto computed_iw1_vv_azimuth_list = alus::tnr::GetAzimuthNoiseVectorList(iw1_vv_azimuth_list);
    test::utils::AssertAzimuthNoiseVectorsAreSame(test::expectedvalues::IW1_VV_NOISE_AZIMUTH_VECTOR_LIST,
                                                  computed_iw1_vv_azimuth_list);

    const auto iw2_vh_azimuth_list = original_metadata_root_->GetElement(alus::snapengine::AbstractMetadata::NOISE)
                                         ->GetElement(test::constants::IW2_VH_DATA.image_name)
                                         ->GetElement(alus::snapengine::AbstractMetadata::NOISE)
                                         ->GetElement(alus::snapengine::AbstractMetadata::NOISE_AZIMUTH_VECTOR_LIST);

    const auto computed_iw2_vh_azimuth_list = alus::tnr::GetAzimuthNoiseVectorList(iw2_vh_azimuth_list);
    test::utils::AssertAzimuthNoiseVectorsAreSame(test::expectedvalues::IW2_VH_NOISE_AZIMUTH_VECTOR_LIST,
                                                  computed_iw2_vh_azimuth_list);
}

TEST_F(ThermalNoiseUtilsTest, getNoiseVectorListTest) {
    const auto iw1_vv_noise_vector_list = original_metadata_root_->GetElement(alus::snapengine::AbstractMetadata::NOISE)
                                              ->GetElement(test::constants::IW1_VV_DATA.image_name)
                                              ->GetElement(alus::snapengine::AbstractMetadata::NOISE)
                                              ->GetElement(alus::snapengine::AbstractMetadata::NOISE_RANGE_VECTOR_LIST);
    const auto computed_iw1_vv_noise_list = alus::tnr::GetNoiseVectorList(iw1_vv_noise_vector_list);
    test::utils::AssertNoiseVectorListsAreSame(test::expectedvalues::IW1_VV_NOISE_VECTOR_LIST,
                                               computed_iw1_vv_noise_list);

    const auto iw2_vh_noise_vector_list = original_metadata_root_->GetElement(alus::snapengine::AbstractMetadata::NOISE)
                                              ->GetElement(test::constants::IW2_VH_DATA.image_name)
                                              ->GetElement(alus::snapengine::AbstractMetadata::NOISE)
                                              ->GetElement(alus::snapengine::AbstractMetadata::NOISE_RANGE_VECTOR_LIST);
    const auto computed_iw2_vh_noise_list = alus::tnr::GetNoiseVectorList(iw2_vh_noise_vector_list);
    test::utils::AssertNoiseVectorListsAreSame(test::expectedvalues::IW2_VH_NOISE_VECTOR_LIST,
                                               computed_iw2_vh_noise_list);
}

TEST_F(ThermalNoiseUtilsTest, fillTimeMapsWithT0AndDeltaTStest) {
    alus::tnr::TimeMaps iw1_vv_time_maps;
    alus::tnr::FillTimeMapsWithT0AndDeltaTS(test::constants::IW1_VV_DATA.image_name, original_metadata_root_,
                                                     iw1_vv_time_maps);
    test::utils::AssertTimeMapsAreSame(test::expectedvalues::IW1_VV_TIME_MAPS, iw1_vv_time_maps);

    alus::tnr::TimeMaps iw2_vh_time_maps;
    alus::tnr::FillTimeMapsWithT0AndDeltaTS(test::constants::IW2_VH_DATA.image_name, original_metadata_root_,
                                                     iw2_vh_time_maps);
    test::utils::AssertTimeMapsAreSame(test::expectedvalues::IW2_VH_TIME_MAPS, iw2_vh_time_maps);
}

TEST_F(ThermalNoiseUtilsTest, getBurstRangeVectorTest) {
    const std::vector<std::pair<int, int>> center_line_to_burst_index{
        {12775, 9}, {751, 1}, {2254, 2}, {5260, 4}, {6763, 5}, {8266, 6}, {9769, 7}, {11272, 8}, {3757, 3},
    };

    for (const auto& [burst_center_line, vector_index] : center_line_to_burst_index) {
        const auto computed_vector =
            alus::tnr::GetBurstRangeVector(burst_center_line, test::constants::NOISE_VECTORS);
        test::utils::AssertNoiseVectorsAreSame(test::constants::NOISE_VECTORS.at(vector_index), computed_vector);
    }
}
}  // namespace