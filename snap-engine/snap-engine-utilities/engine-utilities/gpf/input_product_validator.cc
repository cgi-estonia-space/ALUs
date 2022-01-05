/**
 * This file is a filtered duplicate of a SNAP's
 * org.esa.snap.engine_utilities.gpf.InputProductValidator.java
 * ported for native code.
 * Copied from (https://github.com/senbox-org/snap-engine). It was originally stated:
 *
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
#include "snap-engine-utilities/engine-utilities/gpf/input_product_validator.h"

#include <boost/algorithm/string/predicate.hpp>
#include <boost/algorithm/string/trim.hpp>

#include "snap-core/core/datamodel/band.h"
#include "snap-core/core/datamodel/metadata_attribute.h"
#include "snap-core/core/datamodel/metadata_element.h"
#include "snap-core/core/datamodel/product.h"
#include "snap-core/core/util/string_utils.h"
#include "snap-engine-utilities/engine-utilities/datamodel/metadata/abstract_metadata.h"
#include "snap-engine-utilities/engine-utilities/datamodel/unit.h"
#include "snap-engine-utilities/engine-utilities/gpf/operator_utils.h"

namespace alus::snapengine {

InputProductValidator::InputProductValidator(const std::shared_ptr<Product>& product) {
    product_ = product;
    abs_root_ = AbstractMetadata::GetAbstractedMetadata(product);
}
bool InputProductValidator::IsSARProduct() {
    return abs_root_ != nullptr && abs_root_->GetAttributeDouble("radar_frequency", 99999) != 99999;  // NOLINT
}
void InputProductValidator::CheckIfSARProduct() {
    if ("RAW" == product_->GetProductType()) {
        throw std::runtime_error(std::string(SHOULD_NOT_BE_LEVEL0));
    }
    if (!IsSARProduct()) {
        throw std::runtime_error(std::string(SHOULD_BE_SAR_PRODUCT));
    }
}
bool InputProductValidator::IsComplex() {
    if (abs_root_) {
        std::string sample_type =
            abs_root_->GetAttributeString(AbstractMetadata::SAMPLE_TYPE, AbstractMetadata::NO_METADATA_STRING);
        boost::algorithm::trim(sample_type);
        return boost::iequals(sample_type, "complex");
    }
    return false;
}
void InputProductValidator::CheckIfSLC() {
    if (!IsComplex()) {
        throw std::runtime_error(std::string(SHOULD_BE_SLC));
    }
}
void InputProductValidator::CheckIfGRD() {
    if (IsComplex()) {
        throw std::runtime_error(std::string(SHOULD_BE_GRD));
    }
}
bool InputProductValidator::IsMultiSwath() {
    std::vector<std::string> band_names = product_->GetBandNames();
    return (Contains(band_names, "IW1") && Contains(band_names, "IW2")) ||
           (Contains(band_names, "EW1") && Contains(band_names, "EW2"));
}
bool InputProductValidator::Contains(const std::vector<std::string>& list, std::string_view tag) {
    return std::any_of(std::begin(list), std::end(list),
                       [&tag](const auto& string) { return string.find(tag) != std::string::npos; });
}
bool InputProductValidator::IsSentinel1Product() {
    std::string mission = abs_root_->GetAttributeString(AbstractMetadata::MISSION);
    return (mission.rfind("SENTINEL-1", 0) != std::string::npos);
}
void InputProductValidator::CheckIfSentinel1Product() {
    if (!IsSentinel1Product()) {
        throw std::runtime_error(std::string(SHOULD_BE_S1));
    }
}
void InputProductValidator::CheckMission(const std::vector<std::string>& valid_missions) {
    std::string mission = abs_root_->GetAttributeString(AbstractMetadata::MISSION, "");
    boost::to_upper(mission);
    for (auto valid_mission : valid_missions) {
        boost::to_upper(valid_mission);
        if (mission.rfind(valid_mission, 0) != std::string::npos) {
            return;
        }
    }
    throw std::runtime_error(mission +
                             " is not a valid mission from: " + StringUtils::ArrayToString(valid_missions, ","));
}
void InputProductValidator::CheckProductType(const std::vector<std::string>& valid_product_types) {
    std::string product_type = abs_root_->GetAttributeString(AbstractMetadata::PRODUCT_TYPE, "");
    for (auto const& valid_product_type : valid_product_types) {
        if (boost::equals(product_type, valid_product_type)) {
            return;
        }
    }
    throw std::runtime_error(
        product_type + " is not a valid product type from: " + StringUtils::ArrayToString(valid_product_types, ","));
}
void InputProductValidator::CheckAcquisitionMode(const std::vector<std::string>& valid_modes) {
    std::string acquisition_mode = abs_root_->GetAttributeString(AbstractMetadata::ACQUISITION_MODE);
    for (auto const& valid_mode : valid_modes) {
        if (boost::equals(acquisition_mode, valid_mode)) {
            return;
        }
    }
    throw std::runtime_error(acquisition_mode +
                             " is not a valid acquisition mode from: " + StringUtils::ArrayToString(valid_modes, ","));
}
bool InputProductValidator::IsTOPSARProduct() {
    bool is_s1 = false;
    std::string mission = abs_root_ != nullptr ? abs_root_->GetAttributeString(AbstractMetadata::MISSION, "") : "";
    if ((mission.rfind("SENTINEL-1", 0) != std::string::npos) ||
        (mission.rfind("RS2", 0) != std::string::npos)) {  // also include RS2 in TOPS mode
        is_s1 = true;
    }
    std::vector<std::string> band_names = product_->GetBandNames();
    return is_s1 && (Contains(band_names, "IW1") || Contains(band_names, "IW2") || Contains(band_names, "IW3") ||
                     Contains(band_names, "EW1") || Contains(band_names, "EW2") || Contains(band_names, "EW3") ||
                     Contains(band_names, "EW4") || Contains(band_names, "EW5"));
}
void InputProductValidator::CheckIfTOPSARBurstProduct(bool shouldbe) {
    {
        bool is_topsar_product = IsTOPSARProduct();
        if (shouldbe && !is_topsar_product) {
            // It should be a TOP SAR Burst product, but it is not even a TOP SAR Product
            throw std::runtime_error("Source product should be an SLC burst product");
        }
        if (shouldbe && IsDebursted()) {
            // It should be a TOP SAR Burst product and it is a TOP SAR product but it has been deburst
            throw std::runtime_error("Source product should NOT be a deburst product");
        }
        if (!shouldbe && is_topsar_product && !IsDebursted()) {
            // It should not be a TOP SAR burst product but it is.
            throw std::runtime_error(std::string(SHOULD_BE_DEBURST));
        }
    }
}
void InputProductValidator::CheckIfMultiSwathTOPSARProduct() {
    if (!IsMultiSwath()) {
        throw std::runtime_error(std::string(SHOULD_BE_MULTISWATH_SLC));
    }
}
bool InputProductValidator::IsDebursted() {
    if (!IsSentinel1Product()) return true;

    bool is_debursted = true;
    std::shared_ptr<MetadataElement> orig_prod_root = AbstractMetadata::GetOriginalProductMetadata(product_);
    std::shared_ptr<MetadataElement> annotation = orig_prod_root->GetElement("annotation");
    if (annotation == nullptr) {
        return true;
    }

    std::vector<std::shared_ptr<MetadataElement>> elems = annotation->GetElements();
    for (auto const& elem : elems) {
        auto product = elem->GetElement("product");
        if (product) {
            auto swath_timing = product->GetElement("swathTiming");
            auto burst_list = swath_timing->GetElement("burstList");
            int count = std::stoi(burst_list->GetAttributeString("count"));
            if (count != 0) {
                is_debursted = false;
                break;
            }
        }
    }
    return is_debursted;
}
bool InputProductValidator::IsFullPolSLC() {
    int valid_band_cnt = 0;
    for (auto const& band : product_->GetBands()) {
        UnitType band_unit = Unit::GetUnitType(band);
        if (!(band_unit == UnitType::REAL || band_unit == UnitType::IMAGINARY)) {
            continue;
        }
        std::string pol = OperatorUtils::GetPolarizationFromBandName(band->GetName());
        if (pol.empty()) {
            continue;
        }

        if ((pol.find("hh") != std::string::npos) || (pol.find("hv") != std::string::npos) ||
            (pol.find("vh") != std::string::npos) || (pol.find("vv") != std::string::npos)) {
            ++valid_band_cnt;
        }
    }

    return (valid_band_cnt == 8);  // NOLINT
}
void InputProductValidator::CheckIfQuadPolSLC() {
    if (!IsFullPolSLC()) {
        throw std::runtime_error(std::string(SHOULD_BE_QUAD_POL));
    }
}
bool InputProductValidator::IsCalibrated() {
    return (abs_root_ != nullptr &&
            abs_root_->GetAttribute(AbstractMetadata::ABS_CALIBRATION_FLAG)->GetData()->GetElemBoolean());
}
void InputProductValidator::CheckIfCalibrated(bool should_be) {
    bool is_calibrated = IsCalibrated();
    if (!should_be && is_calibrated) {
        throw std::runtime_error(std::string(SHOULD_NOT_BE_CALIBRATED));
    }
    if (should_be && !is_calibrated) {
        throw std::runtime_error(std::string(SHOULD_BE_CALIBRATED));
    }
}
void InputProductValidator::CheckIfTanDEMXProduct() {
    std::string mission = abs_root_ != nullptr ? abs_root_->GetAttributeString(AbstractMetadata::MISSION) : "";
    if (mission.rfind("TDM", 0) == std::string::npos) {
        throw std::runtime_error("Input should be a TanDEM-X product.");
    }
}

// todo: IsCompatibleProduct needs geocoding (provide if needed)
// void InputProductValidator::CheckIfCompatibleProducts(std::vector<std::shared_ptr<Product>> source_products) {
//    for (auto const& src_product : source_products) {
//        if (!product_->IsCompatibleProduct(src_product, GEOGRAPHIC_ERROR)) {
//            throw std::runtime_error(std::string(SHOULD_BE_COMPATIBLE));
//        }
//    }
//}

// bool InputProductValidator::IsMapProjected(const std::shared_ptr<Product>& product) {
//    if (std::dynamic_pointer_cast<MapGeoCoding>(product->GetSceneGeoCoding()) ||
//        std::dynamic_pointer_cast<CrsGeoCoding>(product->GetSceneGeoCoding())) {
//        return true;
//    }
//    if (MetaDataNodeNames::HasAbstractedMetadata(product)) {
//        std::shared_ptr<MetadataElement> abs_root = MetaDataNodeNames::GetAbstractedMetadata(product);
//        return abs_root != nullptr && !MetaDataNodeNames::IsNoData(abs_root, MetaDataNodeNames::MAP_PROJECTION);
//    }
//    return false;
//}

}  // namespace alus::snapengine
