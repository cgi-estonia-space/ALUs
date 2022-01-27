/**
 * This file is a filtered duplicate of a SNAP's
 * org.esa.snap.core.datamodel.SampleCoding.java
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
#include "sample_coding.h"

#include <stdexcept>

#include "../util/guardian.h"
#include "snap-core/core/datamodel/metadata_attribute.h"

namespace alus::snapengine {

SampleCoding::SampleCoding(std::string_view name) : MetadataElement(name) {}

void SampleCoding::AddElement([[maybe_unused]] const std::shared_ptr<MetadataElement>& element) {
    // just override to prevent add
    throw std::runtime_error("Add element not supported for SampleCoding");
}
void SampleCoding::AddAttribute(const std::shared_ptr<MetadataAttribute>& attribute) {
    if (!attribute->GetData()->IsInt()) {
        throw std::invalid_argument("attribute value is not a integer");
    }
    if (attribute->GetData()->GetNumElems() == 0) {
        throw std::invalid_argument("attribute value is missing");
    }
    if (ContainsAttribute(attribute->GetName())) {
        throw std::invalid_argument("SampleCoding contains already an attribute with the name '" +
                                    std::string(attribute->GetName()) + "'");
    }
    MetadataElement::AddAttribute(attribute);
}

std::shared_ptr<MetadataAttribute> SampleCoding::AddSample(std::string_view name, int value,
                                                           std::string_view description) {
    return AddSamples(name, std::vector<int>(value), description);
}
std::shared_ptr<MetadataAttribute> SampleCoding::AddSamples(std::string_view name, const std::vector<int>& values,
                                                            std::string_view description) {
    Guardian::AssertNotNull("name", name);
    std::shared_ptr<ProductData> product_data =
        ProductData::CreateInstance(ProductData::TYPE_UINT32, static_cast<int>(values.size()));
    std::shared_ptr<MetadataAttribute> attribute = std::make_shared<MetadataAttribute>(name, product_data, false);
    attribute->SetDataElems(values);
    if (description != nullptr) {  // NOLINT
        attribute->SetDescription(description);
    }
    AddAttribute(attribute);
    return attribute;
}
int SampleCoding::GetSampleCount() { return GetNumAttributes(); }
std::string SampleCoding::GetSampleName(int index) { return GetAttributeAt(index)->GetName(); }
int SampleCoding::GetSampleValue(int index) { return GetAttributeAt(index)->GetData()->GetElemInt(); }

}  // namespace alus::snapengine
