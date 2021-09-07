/**
 * This file is a filtered duplicate of a SNAP's
 * org.esa.snap.core.datamodel.MetadataElement.java
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
#include "snap-core/core/datamodel/metadata_element.h"

#include <stdexcept>

#include "../util/guardian.h"
#include "alus_log.h"
#include "product_node_group.h"
#include "snap-core/core/dataio/product_subset_def.h"
#include "snap-core/core/datamodel/metadata_attribute.h"

namespace alus {
namespace snapengine {

MetadataElement::MetadataElement(std::string_view name, std::string_view description, IMetaDataReader* meta_data_reader)
    : ProductNode(name, description) {
    this->meta_data_reader_ = meta_data_reader;
    // reader provides everything needed to construct the element (attributes, elements etc..)
    //    todo:make sure it runs once

    this->elements_ = this->meta_data_reader_->Read(name)->elements_;
    this->attributes_ = this->meta_data_reader_->Read(name)->attributes_;
}

/**
 * Adds the given element to this element.
 *
 * @param element the element to added, ignored if <code>null</code>
 */
void MetadataElement::AddElement(std::shared_ptr<MetadataElement> element) {
    if (element == nullptr) {
        return;
    }
    if (elements_ == nullptr) {
        elements_ = std::make_shared<ProductNodeGroup<std::shared_ptr<MetadataElement>>>(this, "elements", true);
    }
    elements_->Add(element);
}

/**
 * Adds an attribute to this node.
 *
 * @param attribute the attribute to be added, <code>null</code> is ignored
 */
void MetadataElement::AddAttribute(std::shared_ptr<MetadataAttribute> attribute) {
    if (attribute == nullptr) {
        return;
    }
    if (attributes_ == nullptr) {
        attributes_ = std::make_shared<ProductNodeGroup<std::shared_ptr<MetadataAttribute>>>(this, "attributes", true);
    }
    attributes_->Add(attribute);
}

/**
 * @return the number of elements contained in this element.
 */
int MetadataElement::GetNumElements() const {
    if (elements_ == nullptr) {
        return 0;
    }
    return elements_->GetNodeCount();
}

/**
 * Returns the number of attributes attached to this node.
 *
 * @return the number of attributes
 */
int MetadataElement::GetNumAttributes() const {
    if (attributes_ == nullptr) {
        return 0;
    }
    return attributes_->GetNodeCount();
}

bool MetadataElement::RemoveAttribute(std::shared_ptr<MetadataAttribute> attribute) {
    return attribute != nullptr && attributes_ != nullptr && attributes_->Remove(attribute);
}

/**
 * Returns the names of all attributes of this node.
 *
 * @return the attribute name array, never <code>null</code>
 */
std::vector<std::string> MetadataElement::GetAttributeNames() const {
    if (attributes_ == nullptr) {
        return std::vector<std::string>();
    }
    return attributes_->GetNodeNames();
}

bool MetadataElement::ContainsAttribute(std::string_view name) const {
    return attributes_ != nullptr && attributes_->Contains(name);
}

/**
 * Tests if a element with the given name is contained in this element.
 *
 * @param name the name, must not be <code>null</code>
 *
 * @return <code>true</code> if a element with the given name is contained in this element, <code>false</code>
 *         otherwise
 */
bool MetadataElement::ContainsElement(std::string_view name) const {
    Guardian::AssertNotNullOrEmpty("name", name);
    return elements_ != nullptr && elements_->Contains(name);
}

/**
 * Gets the index of the given element.
 *
 * @param element The element .
 *
 * @return The element's index, or -1.
 *
 * @since BEAM 4.7
 */
int MetadataElement::GetElementIndex(const std::shared_ptr<MetadataElement>& element) const {
    return elements_->IndexOf(element);
}

int MetadataElement::GetAttributeIndex(const std::shared_ptr<MetadataAttribute>& attribute) const {
    return attributes_->IndexOf(attribute);
}

int MetadataElement::GetAttributeInt(std::string_view name, int default_value) const {
    auto attribute = GetAttribute(name);
    if (!attribute) {
        return default_value;
    }
    if (attribute->GetDataType() == ProductData::TYPE_ASCII) {
        return std::stoi(attribute->GetData()->GetElemString());
    }
    return attribute->GetData()->GetElemInt();
}

int MetadataElement::GetAttributeInt(std::string_view name) const {
    auto attribute = GetAttribute(name);
    if (!attribute) {
        throw std::invalid_argument(GetAttributeNotFoundMessage(name));
    }
    if (attribute->GetDataType() == ProductData::TYPE_ASCII) {
        return std::stoi(attribute->GetData()->GetElemString());
    }
    return attribute->GetData()->GetElemInt();
}

std::shared_ptr<MetadataAttribute> MetadataElement::GetAttribute(std::string_view name) const {
    if (attributes_ == nullptr) {
        return nullptr;
    }
    return attributes_->Get(name);
}

double MetadataElement::GetAttributeDouble(std::string_view name, double default_value) const {
    auto attribute = GetAttribute(name);
    if (!attribute) {
        return default_value;
    }
    if (attribute->GetDataType() == ProductData::TYPE_ASCII) {
        return std::stod(attribute->GetData()->GetElemString());
    }
    return attribute->GetData()->GetElemDouble();
}

double MetadataElement::GetAttributeDouble(std::string_view name) const {
    auto attribute = GetAttribute(name);
    if (!attribute) {
        throw std::invalid_argument(GetAttributeNotFoundMessage(name));
    }
    if (attribute->GetDataType() == ProductData::TYPE_ASCII) {
        return std::stod(attribute->GetData()->GetElemString());
    }
    return attribute->GetData()->GetElemDouble();
}

std::shared_ptr<Utc> MetadataElement::GetAttributeUtc(std::string_view name, std::shared_ptr<Utc> default_value) const {
    try {
        std::shared_ptr<MetadataAttribute> attribute = GetAttribute(name);
        if (attribute) {
            return Utc::Parse(attribute->GetData()->GetElemString());
        }
    } catch (const std::exception& e) {
        LOGW << e.what();
        // continue
    }
    return default_value;
}
std::shared_ptr<Utc> MetadataElement::GetAttributeUtc(std::string_view name) const {
    try {
        auto attribute = GetAttribute(name);
        if (attribute) {
            return Utc::Parse(attribute->GetData()->GetElemString());
        }
    } catch (const std::runtime_error& e) {
        throw std::invalid_argument("Unable to parse metadata attribute " + std::string{name});
    }
    throw std::invalid_argument(GetAttributeNotFoundMessage(name));
}
std::string MetadataElement::GetAttributeString(std::string_view name, std::string_view default_value) const {
    auto attribute = GetAttribute(name);
    if (!attribute) {
        return std::string{default_value};
    }
    return attribute->GetData()->GetElemString();
}

std::string MetadataElement::GetAttributeString(std::string_view name) const {
    auto attribute = GetAttribute(name);
    if (!attribute) {
        throw std::invalid_argument(GetAttributeNotFoundMessage(name));
    }
    return attribute->GetData()->GetElemString();
}

void MetadataElement::SetAttributeInt(std::string_view name, int value) {
    auto attribute = GetAndMaybeCreateAttribute(name, ProductData::TYPE_INT32, 1);
    attribute->GetData()->SetElemInt(value);
}

void MetadataElement::SetAttributeDouble(std::string_view name, double value) {
    auto attribute = GetAndMaybeCreateAttribute(name, ProductData::TYPE_FLOAT64, 1);
    attribute->GetData()->SetElemDouble(value);
}

void MetadataElement::SetAttributeUtc(std::string_view name, const std::shared_ptr<Utc>& value) {
    auto attribute = GetAndMaybeCreateAttribute(name, ProductData::TYPE_UTC, 1);
    attribute->GetData()->SetElems(value->GetArray());
}

void MetadataElement::SetAttributeString(std::string_view name, std::string_view value) {
    auto attribute = GetAndMaybeCreateAttribute(name, ProductData::TYPE_ASCII, 1);
    attribute->GetData()->SetElems(value);
}

std::shared_ptr<MetadataAttribute> MetadataElement::GetAndMaybeCreateAttribute(std::string_view name, int type,
                                                                               int num_elems) {
    std::shared_ptr<MetadataAttribute> attribute = GetAttribute(name);
    if (!attribute) {
        attribute = std::make_shared<MetadataAttribute>(name, type, num_elems);
        AddAttribute(attribute);
    }
    return attribute;
}

std::shared_ptr<MetadataElement> MetadataElement::CreateDeepClone() {
    auto clone = std::make_shared<MetadataElement>(GetName());
    clone->SetDescription(GetDescription());
    auto attributes = GetAttributes();
    for (const auto& attribute : attributes) {
        clone->AddAttribute(attribute->CreateDeepClone());
    }
    auto elements = GetElements();
    for (const auto& element : elements) {
        clone->AddElement(element->CreateDeepClone());
    }
    return clone;
}

std::string MetadataElement::GetAttributeNotFoundMessage(std::string_view name) {
    return "Metadata attribute '" + std::string{name} + "' not found";
}

std::shared_ptr<MetadataElement> MetadataElement::GetParentElement([[maybe_unused]] const ProductNode& node) {
    // todo:provide implementation and remove "maybe_unused"
    throw std::runtime_error(
        "called not yet implemented method MetadataElement::GetParentElement(const ProductNode& node)");
}

std::shared_ptr<MetadataElement> MetadataElement::GetElement(std::string_view name) {
    Guardian::AssertNotNullOrEmpty("name", name);
    if (elements_ == nullptr) {
        return nullptr;
    }
    return elements_->Get(name);
}

std::vector<std::shared_ptr<MetadataElement>> MetadataElement::GetElements() {
    if (elements_ == nullptr) {
        return std::vector<std::shared_ptr<MetadataElement>>();
    }
    return elements_->ToArray(std::vector<std::shared_ptr<MetadataElement>>(elements_->GetNodeCount()));
}

std::vector<std::shared_ptr<MetadataAttribute>> MetadataElement::GetAttributes() {
    if (attributes_ == nullptr) {
        return std::vector<std::shared_ptr<MetadataAttribute>>();
    }
    return attributes_->ToArray(std::vector<std::shared_ptr<MetadataAttribute>>(attributes_->GetNodeCount()));
}

void MetadataElement::AddElementAt(const std::shared_ptr<MetadataElement>& element, int index) {
    if (element == nullptr) {
        return;
    }
    if (elements_ == nullptr) {
        elements_ = std::make_shared<ProductNodeGroup<std::shared_ptr<MetadataElement>>>(this, "elements", true);
    }
    elements_->Add(index, element);
}

bool MetadataElement::RemoveElement(const std::shared_ptr<MetadataElement>& element) {
    return element != nullptr && elements_ != nullptr && elements_->Remove(element);
}

std::shared_ptr<MetadataAttribute> MetadataElement::GetAttributeAt(int index) const {
    if (attributes_ == nullptr) {
        throw std::runtime_error("index out of bounds exception");
    }
    return attributes_->Get(index);
}
uint64_t MetadataElement::GetRawStorageSize(const std::shared_ptr<ProductSubsetDef>& subset_def) {
    if (subset_def != nullptr && !subset_def->ContainsNodeName(GetName())) {
        return 0L;
    }
    uint64_t size = 0;
    for (int i = 0; i < GetNumElements(); i++) {
        size += GetElementAt(i)->GetRawStorageSize(subset_def);
    }
    for (int i = 0; i < GetNumAttributes(); i++) {
        size += GetAttributeAt(i)->GetRawStorageSize(subset_def);
    }
    return size;
}
std::shared_ptr<MetadataElement> MetadataElement::GetElementAt(int index) {
    if (elements_ == nullptr) {
        throw std::runtime_error("no elements available at index: " + std::to_string(index));
    }
    return elements_->Get(index);
}
void MetadataElement::Dispose() {
    if (attributes_) {
        attributes_->Dispose();
        attributes_ = nullptr;
    }
    if (elements_) {
        elements_->Dispose();
        elements_ = nullptr;
    }
    ProductNode::Dispose();
}
std::vector<std::string> MetadataElement::GetElementNames() const {
    return elements_ ? elements_->GetNodeNames() : std::vector<std::string>(0);
}

}  // namespace snapengine
}  // namespace alus
