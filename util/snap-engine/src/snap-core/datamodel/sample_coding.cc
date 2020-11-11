#include "sample_coding.h"

#include <stdexcept>

#include "guardian.h"
#include "snap-core/datamodel/metadata_attribute.h"

namespace alus {
namespace snapengine {

SampleCoding::SampleCoding(std::string_view name) : MetadataElement(name) {}

void SampleCoding::AddElement([[maybe_unused]] std::shared_ptr<MetadataElement> element) {
    // just override to prevent add
    throw std::runtime_error("Add element not supported for SampleCoding");
}
void SampleCoding::AddAttribute(std::shared_ptr<MetadataAttribute> attribute) {
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
std::shared_ptr<MetadataAttribute> SampleCoding::AddSamples(std::string_view name, std::vector<int> values,
                                                            std::string_view description) {
    Guardian::AssertNotNull("name", name);
    std::shared_ptr<ProductData> product_data = ProductData::CreateInstance(ProductData::TYPE_UINT32, values.size());
    std::shared_ptr<MetadataAttribute> attribute = std::make_shared<MetadataAttribute>(name, product_data, false);
    attribute->SetDataElems(values);
    if (description != nullptr) {
        attribute->SetDescription(description);
    }
    AddAttribute(attribute);
    return attribute;
}
int SampleCoding::GetSampleCount() { return GetNumAttributes(); }
std::string SampleCoding::GetSampleName(int index) { return GetAttributeAt(index)->GetName(); }
int SampleCoding::GetSampleValue(int index) { return GetAttributeAt(index)->GetData()->GetElemInt(); }

}  // namespace snapengine
}  // namespace alus
