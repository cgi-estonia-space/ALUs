/**
 * This file is a filtered duplicate of a SNAP's
 * org.esa.snap.engine_utilities.gpf.StackUtils.java
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

#include "stack_utils.h"

#include <memory>
#include <string>

#include <boost/algorithm/string.hpp>

#include "snap-core/core//datamodel/metadata_element.h"
#include "snap-core/core/datamodel/product.h"
#include "snap-core/core/util/string_utils.h"
#include "snap-engine-utilities/engine-utilities/datamodel/metadata/abstract_metadata.h"
#include "snap-engine-utilities/engine-utilities/gpf/operator_utils.h"

namespace alus::snapengine {

std::string StackUtils::CreateBandTimeStamp(std::shared_ptr<Product>& product) {
    std::shared_ptr<MetadataElement> abs_root = AbstractMetadata::GetAbstractedMetadata(product);
    if (abs_root != nullptr) {
        std::string date_string = OperatorUtils::GetAcquisitionDate(abs_root);
        if (!date_string.empty()) date_string = '_' + date_string;
        return StringUtils::CreateValidName(date_string, "_.", '_');
    }
    return "";
}

bool StackUtils::IsMasterBand(std::string_view band_name, const std::shared_ptr<Product>& product) {
    if (const auto slave_metadata_root = product->GetMetadataRoot()->GetElement(AbstractMetadata::SLAVE_METADATA_ROOT);
        slave_metadata_root) {
        const auto master_band_names = slave_metadata_root->GetAttributeString(AbstractMetadata::MASTER_BANDS, "");
        return master_band_names.find(band_name) != std::string::npos;
    }

    return std::any_of(product->GetBandNames().begin(), product->GetBandNames().end(),
                       [&band_name](std::string_view source_band_name) {
                           return boost::algorithm::icontains(source_band_name, MST) &&
                                  boost::algorithm::contains(source_band_name, band_name);
                       });
}
bool StackUtils::IsSlaveBand(std::string_view band_name, const std::shared_ptr<Product> product) {
    const auto slave_band_names = GetSlaveProductNames(product);
    return std::any_of(slave_band_names.begin(), slave_band_names.end(),
                       [&band_name, &product](std::string_view slave_product_name) {
                           return IsSlaveBand(band_name, product, slave_product_name);
                       });
}
std::vector<std::string> StackUtils::GetSlaveProductNames(const std::shared_ptr<Product>& product) {
    const auto slave_metadata_root = product->GetMetadataRoot()->GetElement(AbstractMetadata::SLAVE_METADATA_ROOT);
    return slave_metadata_root ? slave_metadata_root->GetElementNames() : std::vector<std::string>(0);
}
bool StackUtils::IsSlaveBand(std::string_view band_name, const std::shared_ptr<Product>& product,
                             std::string_view slave_product_name) {
    if (const auto slave_metadata_root = product->GetMetadataRoot()->GetElement(AbstractMetadata::SLAVE_METADATA_ROOT);
        slave_metadata_root) {
        const auto element = slave_metadata_root->GetElement(slave_product_name);
        const auto slave_band_names = element->GetAttributeString(AbstractMetadata::SLAVE_BANDS, "");
        if (!slave_band_names.empty()) {
            return boost::algorithm::contains(slave_band_names, band_name);
        }
    }

    const auto date_suffix = slave_product_name.substr(slave_product_name.rfind('_'));

    return std::any_of(product->GetBandNames().begin(), product->GetBandNames().end(),
                       [&band_name, &date_suffix](std::string_view source_band_name) {
                           return boost::algorithm::icontains(source_band_name, SLV) &&
                                  boost::algorithm::iends_with(source_band_name, date_suffix) &&
                                  boost::algorithm::contains(source_band_name, band_name);
                       });
}
bool StackUtils::IsCoregisteredStack(const std::shared_ptr<Product>& product) {
    if (AbstractMetadata::HasAbstractedMetadata(product)) {
        const auto abs_root = AbstractMetadata::GetAbstractedMetadata(product);
        return abs_root != nullptr && abs_root->GetAttributeInt(AbstractMetadata::COREGISTERED_STACK, 0) == 1;
    }

    return false;
}

}  // namespace alus::snapengine
