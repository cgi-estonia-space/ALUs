#include "index_coding.h"

#include "guardian.h"
#include "metadata_attribute.h"

namespace alus {
namespace snapengine {

IndexCoding::IndexCoding(std::string_view name) : SampleCoding(name) {}
std::shared_ptr<MetadataAttribute> IndexCoding::GetIndex(std::string_view name) { return GetAttribute(name); }
std::vector<std::string> IndexCoding::GetIndexNames() { return GetAttributeNames(); }
std::shared_ptr<MetadataAttribute> IndexCoding::AddIndex(std::string_view name, int value,
                                                         std::string_view description) {
    return AddSample(name, value, description);
}
int IndexCoding::GetIndexValue(std::string_view name) {
    Guardian::AssertNotNull("name", name);
    std::shared_ptr<MetadataAttribute> attribute = GetAttribute(name);
    if (attribute == nullptr) {
        throw std::invalid_argument("index '" + std::string(name) + "' not found");
    }
    //    Debug::AssertTrue(attribute->GetData()->IsInt());
    //    Debug::AssertTrue(attribute->GetData()->IsScalar());
    return attribute->GetData()->GetElemInt();
}
}  // namespace snapengine
}  // namespace alus
