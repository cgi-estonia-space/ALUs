/**
 * This file is a filtered duplicate of a SNAP's
 * org.esa.s1tbx.commons.test.MetadataValidator.java
 * ported for native code.
 * Copied from(https://github.com/senbox-org/s1tbx). It was originally stated:
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
#include "s1tbx-commons/test/metadata_validator.h"

#include <stdexcept>

#include <boost/algorithm/string/trim.hpp>

#include "snap-core/core/datamodel/metadata_element.h"
#include "snap-engine-utilities/engine-utilities/datamodel/metadata/abstract_metadata.h"
#include "snap-engine-utilities/engine-utilities/gpf/input_product_validator.h"

namespace alus::s1tbx {

MetadataValidator::MetadataValidator(const std::shared_ptr<snapengine::Product>& product)
    : MetadataValidator(product, nullptr) {}

MetadataValidator::MetadataValidator(const std::shared_ptr<snapengine::Product>& product,
                                     const std::shared_ptr<ValidationOptions>& options)
    : product_(product),
      abs_root_(snapengine::AbstractMetadata::GetAbstractedMetadata(product)),
      validation_options_(options == nullptr ? std::make_shared<ValidationOptions>() : options) {}

void MetadataValidator::Validate() {
    VerifyStr(snapengine::AbstractMetadata::PRODUCT);
    VerifyStr(snapengine::AbstractMetadata::PRODUCT_TYPE);
    VerifyStr(snapengine::AbstractMetadata::MISSION);
    VerifyStr(snapengine::AbstractMetadata::ACQUISITION_MODE);
    VerifyStr(snapengine::AbstractMetadata::SPH_DESCRIPTOR);
    VerifyStr(snapengine::AbstractMetadata::PASS);

    const auto validator = std::make_shared<snapengine::InputProductValidator>(product_);
    if (validator->IsSARProduct()) {
        ValidateSAR();
    } else {
        ValidateOptical();
    }
}

void MetadataValidator::ValidateOptical() { throw std::runtime_error("Is this SAR"); }

void MetadataValidator::ValidateSAR() {
    {
        VerifyStr(snapengine::AbstractMetadata::SAMPLE_TYPE, std::vector<std::string>{"COMPLEX", "DETECTED"});
        VerifyStr(snapengine::AbstractMetadata::ANTENNA_POINTING, std::vector<std::string>{"right", "left"});

        VerifyDouble(snapengine::AbstractMetadata::RADAR_FREQUENCY);
        // verifyDouble(AbstractMetadata.pulse_repetition_frequency);
        // verifyDouble(AbstractMetadata.range_spacing);
        // verifyDouble(AbstractMetadata.azimuth_spacing);
        // verifyDouble(AbstractMetadata.range_looks);
        // verifyDouble(AbstractMetadata.azimuth_looks);
        VerifyDouble(snapengine::AbstractMetadata::LINE_TIME_INTERVAL);
        // verifyDouble(AbstractMetadata.slant_range_to_first_pixel);

        VerifyInt(snapengine::AbstractMetadata::NUM_OUTPUT_LINES);
        VerifyInt(snapengine::AbstractMetadata::NUM_SAMPLES_PER_LINE);

        VerifyUTC(snapengine::AbstractMetadata::STATE_VECTOR_TIME);

        // verifySRGR();
        VerifyOrbitStateVectors();
        // verifyDopplerCentroids();
    }
}

void MetadataValidator::VerifySRGR() {
    const std::shared_ptr<snapengine::MetadataElement> srgr_elem =
        abs_root_->GetElement(snapengine::AbstractMetadata::SRGR_COEFFICIENTS);
    if (srgr_elem) {
        std::vector<std::shared_ptr<snapengine::MetadataElement>> elems = srgr_elem->GetElements();
        if (elems.size() == 0) {
            throw std::runtime_error("SRGR Coefficients not found");
        }
        auto coef_list = elems.at(0);
        if (!coef_list->ContainsAttribute(snapengine::AbstractMetadata::SRGR_COEF_TIME)) {
            throw std::runtime_error("SRGR " + std::string(snapengine::AbstractMetadata::SRGR_COEF_TIME) +
                                     " not found");
        }
        if (!coef_list->ContainsAttribute(snapengine::AbstractMetadata::GROUND_RANGE_ORIGIN)) {
            throw std::runtime_error("SRGR " + std::string(snapengine::AbstractMetadata::GROUND_RANGE_ORIGIN) +
                                     " not found");
        }

        std::vector<std::shared_ptr<snapengine::MetadataElement>> srgr_list = coef_list->GetElements();
        if (srgr_list.size() == 0) {
            throw std::runtime_error("SRGR Coefficients not found");
        }
        auto srgr = srgr_list.at(0);
        if (!srgr->ContainsAttribute(snapengine::AbstractMetadata::SRGR_COEF)) {
            throw std::runtime_error("SRGR " + std::string(snapengine::AbstractMetadata::SRGR_COEF) + " not found");
        }
    } else {
        throw std::runtime_error("SRGR Coefficients not found");
    }
}

void MetadataValidator::VerifyOrbitStateVectors() {
    if (!validation_options_->validate_orbit_state_vectors_) {
        return;
    }
    const std::shared_ptr<snapengine::MetadataElement> orbit_elem =
        abs_root_->GetElement(snapengine::AbstractMetadata::ORBIT_STATE_VECTORS);
    if (orbit_elem) {
        std::vector<std::shared_ptr<snapengine::MetadataElement>> elems = orbit_elem->GetElements();
        if (elems.size() == 0) {
            throw std::runtime_error("Orbit State Vectors not found");
        }
    } else {
        throw std::runtime_error("Orbit State Vectors not found");
    }
}

void MetadataValidator::VerifyDopplerCentroids() {
    const std::shared_ptr<snapengine::MetadataElement> dop_elem =
        abs_root_->GetElement(snapengine::AbstractMetadata::DOP_COEFFICIENTS);
    if (dop_elem) {
        std::vector<std::shared_ptr<snapengine::MetadataElement>> elems = dop_elem->GetElements();
        if (elems.size() == 0) {
            throw std::runtime_error("Doppler Centroids not found");
        }
        auto coef_list = elems.at(0);
        if (!coef_list->ContainsAttribute(snapengine::AbstractMetadata::DOP_COEF_TIME)) {
            throw std::runtime_error("Doppler Centroids " + std::string(snapengine::AbstractMetadata::DOP_COEF_TIME) +
                                     " not found");
        }
        if (!coef_list->ContainsAttribute(snapengine::AbstractMetadata::SLANT_RANGE_TIME)) {
            throw std::runtime_error("Doppler Centroids " +
                                     std::string(snapengine::AbstractMetadata::SLANT_RANGE_TIME) + " not found");
        }
    } else {
        throw std::runtime_error("Doppler Centroids not found");
    }
}

void MetadataValidator::VerifyStr(std::string_view tag) {
    std::string value = abs_root_->GetAttributeString(tag);
    boost::algorithm::trim(value);
    if (value.empty() || value == snapengine::AbstractMetadata::NO_METADATA_STRING) {
        throw std::runtime_error("Metadata " + std::string(tag) + " is invalid " + value);
    }
}

void MetadataValidator::VerifyStr(std::string_view tag, std::vector<std::string> allowed_str) {
    VerifyStr(tag);
    std::string value = abs_root_->GetAttributeString(tag);
    for (const auto& allowed : allowed_str) {
        if (value == allowed) {
            return;
        }
    }
    throw std::runtime_error("Metadata " + std::string(tag) + " is invalid " + value);
}

void MetadataValidator::VerifyDouble(std::string_view tag) {
    double value = abs_root_->GetAttributeDouble(tag);
    if (value == static_cast<double>(snapengine::AbstractMetadata::NO_METADATA)) {
        throw std::runtime_error("Metadata " + std::string(tag) + " is invalid " + std::to_string(value));
    }
}

void MetadataValidator::VerifyInt(std::string_view tag) {
    auto value = abs_root_->GetAttributeInt(tag);
    if (value == snapengine::AbstractMetadata::NO_METADATA) {
        throw std::runtime_error("Metadata " + std::string(tag) + " is invalid " + std::to_string(value));
    }
}

void MetadataValidator::VerifyUTC(std::string_view tag) {
    std::shared_ptr<snapengine::Utc> value = abs_root_->GetAttributeUtc(tag);
    if (value == nullptr || value == snapengine::AbstractMetadata::NO_METADATA_UTC) {
        throw std::runtime_error("Metadata " + std::string(tag) + " is invalid " + value->GetElemString());
    }
}

}  // namespace alus::s1tbx
