
/**
 * This file is a filtered duplicate of a SNAP's
 * org.esa.snap.engine_utilities.gpf.OperatorUtils.java
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
#include "snap-engine-utilities/engine-utilities/gpf/operator_utils.h"

#include <memory>
#include <stdexcept>
#include <vector>

#include <boost/algorithm/string/case_conv.hpp>

#include "alus_log.h"
#include "snap-core/core/datamodel/band.h"
#include "snap-core/core/datamodel/metadata_element.h"
#include "snap-core/core/datamodel/product.h"
#include "snap-core/core/datamodel/product_data_utc.h"
#include "snap-core/core/datamodel/virtual_band.h"
#include "snap-engine-utilities/engine-utilities/datamodel/metadata/abstract_metadata.h"

namespace alus::snapengine {

std::string OperatorUtils::GetPolarizationFromBandName(std::string_view band_name) {
    // Account for possibilities like "x_HH_dB" or "x_HH_times_VV_conj"
    // where the last one will return an exception because it appears to contain
    // multiple polarizations
    std::string pol;
    std::string band_name_str(band_name);
    boost::algorithm::to_lower(band_name_str);

    if (band_name_str.find("_hh") != std::string::npos) {
        pol += "hh";
    }
    if (band_name_str.find("_vv") != std::string::npos) {
        pol += "vv";
    }
    if (band_name_str.find("_hv") != std::string::npos) {
        pol += "hv";
    }
    if (band_name_str.find("_vh") != std::string::npos) {
        pol += "vh";
    }

    // compact pol
    if (band_name_str.find("_rh") != std::string::npos) {
        pol += "rh";
    }
    if (band_name_str.find("_rv") != std::string::npos) {
        pol += "rv";
    }
    if (band_name_str.find("_rch") != std::string::npos) {
        pol += "rch";
    }
    if (band_name_str.find("_rcv") != std::string::npos) {
        pol += "rcv";
    }

    if (pol.length() >= 2 && pol.length() <= 3) {  // NOLINT
        return pol;
    }
    if (pol.length() > 3) {  // NOLINT
        throw std::runtime_error("Band name contains multiple polarizations: " + pol);
    }
    return "";
}

std::vector<std::shared_ptr<Band>> OperatorUtils::GetSourceBands(const std::shared_ptr<Product>& source_product,
                                                                 std::vector<std::string> source_band_names,
                                                                 bool include_virtual_bands) {
    if (source_band_names.empty()) {
        const auto bands = source_product->GetBands();
        std::vector<std::string> band_name_list;
        band_name_list.reserve(source_product->GetNumBands());
        for (const auto& band : bands) {
            // This abomination is used as Java's instanceof.
            if (dynamic_cast<VirtualBand*>(band.get()) == nullptr || include_virtual_bands) {
                band_name_list.push_back(band->GetName());
            }
        }
        source_band_names = band_name_list;
    }

    std::vector<std::shared_ptr<Band>> source_band_list;
    source_band_list.reserve(source_band_names.size());
    for (const auto& source_band_name : source_band_names) {
        const auto source_band = source_product->GetBand(source_band_name);
        if (source_band) {
            source_band_list.push_back(source_band);
        }
    }
    return source_band_list;
}
std::string OperatorUtils::GetSuffixFromBandName(std::string_view band_name) {
    auto index = band_name.find('_');
    if (index != std::string::npos) {
        ++index;
        return std::string(band_name.substr(index));
    }
    index = band_name.find('-');
    if (index != std::string::npos) {
        ++index;
        return std::string(band_name.substr(index));
    }
    index = band_name.find('.');
    if (index != std::string::npos) {
        ++index;
        return std::string(band_name.substr(index));
    }
    return "";
}

std::string OperatorUtils::GetAcquisitionDate(std::shared_ptr<MetadataElement>& root) {
    std::string date_string;

    std::shared_ptr<Utc> date = root->GetAttributeUtc(AbstractMetadata::FIRST_LINE_TIME);
    date_string = date->Format("DDmmmYYYY");
    LOGV << "created date: " << date_string;

    return date_string;
}

}  // namespace alus::snapengine