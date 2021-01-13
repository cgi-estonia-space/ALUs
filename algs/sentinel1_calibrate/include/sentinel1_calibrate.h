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

#include <memory>
#include <set>
#include <string_view>
#include <vector>

#include "calibration_info.h"
#include "metadata_element.h"

namespace alus::sentinel1calibrate {

struct SelectedCalibrationBands {
    bool get_sigma_lut;
    bool get_beta_lut;
    bool get_gamma_lut;
    bool get_dn_lut;
};

/**
 * This is a port of SNAP's Sentinel1Calibrator.getCalibrationVectors(). The name was changed in order to better
 * represent the functions return type;
 *
 * @param original_product_metadata "Original_Product_Metadata" MetadataElement
 * @param selected_polarisations All polarisations for which calibration info should be returned.
 * @param selected_calibration_bands All calibration bands for which calibration info should be returned.
 * @return List of CalibrationInfo structs contained in the given original_product_metadata element.
 */
std::vector<CalibrationInfo> GetCalibrationInfoList(
    const std::shared_ptr<snapengine::MetadataElement>& original_product_metadata,
    std::set<std::string_view> selected_polarisations, SelectedCalibrationBands selected_calibration_bands);

int GetNumOfLines(const std::shared_ptr<snapengine::MetadataElement>& original_product_root, std::string_view polarisation,
                  std::string_view swath);

/**
 * Custom wrapper for MetadataElement::GetElement() function. It checks whether the returned element exists and throws
 * an std::runtime_error if not.
 *
 * @param parent_element Element, whose child should be found.
 * @param element_name Name of the child to be found.
 * @throws std::runtime_element Throws this error if the element is not found.
 * @return
 */
std::shared_ptr<snapengine::MetadataElement> GetElement(const std::shared_ptr<snapengine::MetadataElement>& parent_element,
                                                        std::string_view element_name);
}  // namespace alus::sentinel1calibrate