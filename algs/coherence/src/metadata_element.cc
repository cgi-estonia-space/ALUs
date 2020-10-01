#include "metadata_element.h"

#include "tensorflow/core/platform/default/logging.h"

#include "guardian.h"

namespace alus {
namespace snapengine {

MetadataElement::MetadataElement(std::string_view name, std::string_view description, IMetaDataReader *meta_data_reader)
    : ProductNode(name, description) {
    this->meta_data_reader_ = meta_data_reader;
    // reader provides everything needed to construct the element (attributes, elements etc..)
    //    todo:make sure it runs once

    this->elements_ = this->meta_data_reader_->GetElement(name).elements_;
    this->attributes_ = this->meta_data_reader_->GetElement(name).attributes_;
}
MetadataElement::MetadataElement(std::string_view name,
                                 std::vector<std::shared_ptr<MetadataElement>> elements,
                                 std::vector<std::shared_ptr<MetadataAttribute>> attributes)
    : ProductNode(name) {
    this->elements_ = elements;
    this->attributes_ = attributes;
}
void MetadataElement::AddElement(const std::shared_ptr<MetadataElement> &me) { this->elements_.emplace_back(me); }
void MetadataElement::AddAttribute(const std::shared_ptr<MetadataAttribute> &ma) {
    if (ma == nullptr) {
        return;
    }
    this->attributes_.emplace_back(ma);
}
int MetadataElement::GetNumElements() const { return elements_.size(); }

int MetadataElement::GetNumAttributes() const { return attributes_.size(); }

bool MetadataElement::RemoveAttribute(std::shared_ptr<MetadataAttribute> &ma) {
    auto it =
        std::remove_if(attributes_.begin(), attributes_.end(), [ma](const std::shared_ptr<MetadataAttribute> obj) {
            // todo:might need element wise equality check or name equality check
            return (*obj).Equals(*ma);
        });
    bool any_change = it != attributes_.end();
    attributes_.erase(it, attributes_.end());
    return any_change;
}
std::vector<std::string> MetadataElement::GetAttributeNames() const {
    std::vector<std::string> out(attributes_.size());
    std::transform(
        attributes_.begin(), attributes_.end(), out.begin(), [](const std::shared_ptr<MetadataAttribute> &obj) {
            return obj->GetName();
        });
    return out;
}
bool MetadataElement::ContainsAttribute(std::string_view name) const {
    // check if contains attribute with name...
    return (find_if(attributes_.begin(), attributes_.end(), [&name](const std::shared_ptr<MetadataAttribute> &obj) {
                return obj->GetName() == name;
            }) != attributes_.end());
}
auto MetadataElement::ContainsElement(std::string_view name) const {
    Guardian::AssertNotNullOrEmpty("name", name);
    return (find_if(elements_.begin(), elements_.end(), [&name](const std::shared_ptr<MetadataElement> &obj) {
                return obj->GetName() == name;
            }) != elements_.end());
}

// todo:test
int MetadataElement::GetElementIndex(const std::shared_ptr<MetadataElement> &element) const {
    auto it = std::find(elements_.begin(), elements_.end(), element);
    if (it != elements_.end()) {
        return distance(elements_.begin(), it);
    } else {
        return -1;
    }
}
int MetadataElement::GetAttributeIndex(const std::shared_ptr<MetadataAttribute> &attribute) const {
    auto it = std::find(attributes_.begin(), attributes_.end(), attribute);
    if (it != attributes_.end()) {
        return distance(attributes_.begin(), it);
    } else {
        return -1;
    }
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
    auto it = find_if(attributes_.begin(), attributes_.end(), [name](const std::shared_ptr<MetadataAttribute> &obj) {
        return obj->GetName() == name;
    });
    if (it != attributes_.end()) {
        return *it;
    } else {
        return std::shared_ptr<MetadataAttribute>(nullptr);
    }
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

const std::shared_ptr<Utc> MetadataElement::GetAttributeUtc(std::string_view name,
                                                            std::shared_ptr<Utc> default_value) const {
    try {
        std::shared_ptr<MetadataAttribute> attribute = GetAttribute(name);
        if (attribute) {
            return Utc::Parse(attribute->GetData()->GetElemString());
        }
    } catch (std::exception &e) {
        LOG(INFO) << e.what();
        // continue
    }
    return default_value;
}
const std::shared_ptr<Utc> MetadataElement::GetAttributeUtc(std::string_view name) const {
    try {
        auto attribute = GetAttribute(name);
        if (attribute) {
            return Utc::Parse(attribute->GetData()->GetElemString());
        }
    } catch (std::runtime_error &e) {
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

void MetadataElement::SetAttributeUTC(std::string_view name, const Utc &value) {
    auto attribute = GetAndMaybeCreateAttribute(name, ProductData::TYPE_UTC, 1);
    attribute->GetData()->SetElems(value.GetArray());
}

void MetadataElement::SetAttributeString(std::string_view name, std::string_view value) {
    auto attribute = GetAndMaybeCreateAttribute(name, ProductData::TYPE_ASCII, 1);
    attribute->GetData()->SetElems(value);
}

std::shared_ptr<MetadataAttribute> MetadataElement::GetAndMaybeCreateAttribute(std::string_view name,
                                                                               int type,
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
    for (const auto &attribute : attributes) {
        clone->AddAttribute(attribute->CreateDeepClone());
    }
    auto elements = GetElements();
    for (const auto &element : elements) {
        clone->AddElement(element->CreateDeepClone());
    }
    return clone;
}

std::string MetadataElement::GetAttributeNotFoundMessage(std::string_view name) {
    return "Metadata attribute '" + std::string{name} + "' not found";
}

// todo:provide implementation
std::shared_ptr<MetadataElement> MetadataElement::GetParentElement(const ProductNode &node) {
    return std::shared_ptr<MetadataElement>();
}
std::shared_ptr<MetadataElement> MetadataElement::GetElement(std::string_view name) {
    Guardian::AssertNotNullOrEmpty("name", name);
    auto it = find_if(elements_.begin(), elements_.end(), [name](const std::shared_ptr<MetadataElement> &obj) {
        return obj->GetName() == name;
    });
    if (it != elements_.end()) {
        return *it;
    } else {
        return std::shared_ptr<MetadataElement>(nullptr);
    }
}

// std::shared_ptr<MetadataElement> MetadataElement::GetParentElement(const ProductNode* node) {
//    auto node = node->GetOwner();
//    return node->GetOwner().get();
//    while (node != nullptr) {
//        if (node instanceof MetadataElement) {
//            return (MetadataElement) node;
//        }
//        node = node.getOwner();
//    }
//    return nullptr;
//}

}  // namespace snapengine
}  // namespace alus
