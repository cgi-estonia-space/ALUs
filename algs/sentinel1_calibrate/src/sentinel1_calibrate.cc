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
#include "sentinel1_calibrate.h"

#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <boost/algorithm/string.hpp>

#include "abstract_metadata.h"
#include "calibration_info.h"
#include "general_constants.h"
#include "metadata_element.h"
#include "sentinel1_utils.h"

namespace alus::sentinel1calibrate {
std::vector<CalibrationInfo> GetCalibrationInfoList(
    const std::shared_ptr<snapengine::MetadataElement>& original_product_metadata,
    std::set<std::string_view> selected_polarisations, SelectedCalibrationBands selected_calibration_bands) {
    std::vector<CalibrationInfo> calibration_info_list;

    const auto calibration_root_element =
        GetElement(original_product_metadata, snapengine::AbstractMetadata::CALIBRATION_ROOT);

    for (auto&& calibration_data_set_item : calibration_root_element->GetElements()) {
        const auto calibration_element =
            GetElement(calibration_data_set_item, snapengine::AbstractMetadata::CALIBRATION);

        const auto ads_header_element = GetElement(calibration_element, snapengine::AbstractMetadata::ADS_HEADER);

        const auto polarisation = ads_header_element->GetAttributeString(snapengine::AbstractMetadata::POLARISATION);
        if (selected_polarisations.find(polarisation) == selected_polarisations.end()) {
            continue;
        }

        const auto sub_swath = ads_header_element->GetAttributeString(snapengine::AbstractMetadata::swath);
        const auto first_line_time =
            s1tbx::Sentinel1Utils::GetTime(ads_header_element, snapengine::AbstractMetadata::START_TIME)->GetMjd();
        const auto last_line_time =
            s1tbx::Sentinel1Utils::GetTime(ads_header_element, snapengine::AbstractMetadata::STOP_TIME)->GetMjd();

        const auto num_of_lines = GetNumOfLines(original_product_metadata, polarisation, sub_swath);

        const auto line_time_interval = (last_line_time - first_line_time) / (num_of_lines - 1);

        const auto calibration_vector_list_element =
            GetElement(calibration_element, snapengine::AbstractMetadata::CALIBRATION_VECTOR_LIST);

        const auto count = calibration_vector_list_element->GetAttributeInt(snapengine::AbstractMetadata::COUNT);

        auto calibration_vectors = s1tbx::Sentinel1Utils::GetCalibrationVectors(
            calibration_vector_list_element, selected_calibration_bands.get_sigma_lut,
            selected_calibration_bands.get_beta_lut, selected_calibration_bands.get_gamma_lut,
            selected_calibration_bands.get_dn_lut);

        if (static_cast<size_t>(count) != calibration_vectors.size()) {
            throw std::runtime_error("Invalid amount of calibration vectors in " +
                                     calibration_data_set_item->GetName());
        }

        calibration_info_list.push_back({sub_swath, polarisation, first_line_time, last_line_time, line_time_interval,
                                         num_of_lines, count, calibration_vectors});
    }

    return calibration_info_list;
}
int GetNumOfLines(const std::shared_ptr<snapengine::MetadataElement>& original_product_root,
                  std::string_view polarisation, std::string_view swath) {
    const auto annotation_element = GetElement(original_product_root, snapengine::AbstractMetadata::ANNOTATION);
    for (auto&& annotation_data_set_item : annotation_element->GetElements()) {
        const auto element_name = annotation_data_set_item->GetName();
        if (boost::icontains(element_name, swath) && boost::icontains(element_name, polarisation)) {
            const auto product_element = GetElement(annotation_data_set_item, snapengine::AbstractMetadata::product);
            const auto image_annotation_element =
                GetElement(product_element, snapengine::AbstractMetadata::IMAGE_ANNOTATION);
            const auto image_information_element =
                GetElement(image_annotation_element, snapengine::AbstractMetadata::IMAGE_INFORMATION);
            return image_information_element->GetAttributeInt(snapengine::AbstractMetadata::NUMBER_OF_LINES);
        }
    }

    return snapengine::constants::INVALID_INDEX;
}

std::shared_ptr<snapengine::MetadataElement> GetElement(
    const std::shared_ptr<snapengine::MetadataElement>& parent_element, std::string_view element_name) {
    auto check_that_metadata_exists = [](std::shared_ptr<snapengine::MetadataElement>& element,
                                         std::string_view parent_element, std::string_view element_name) {
        if (!element) {
            throw std::runtime_error(std::string(parent_element) + " is missing " + std::string(element_name) +
                                     " metadata element");
        }
    };

    auto element = parent_element->GetElement(element_name);
    check_that_metadata_exists(element, parent_element->GetName(), element_name);

    return element;
}
}  // namespace alus::sentinel1calibrate
