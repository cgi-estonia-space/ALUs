#include "flag_coding.h"

#include <stdexcept>

#include "guardian.h"

namespace alus {
namespace snapengine {

FlagCoding::FlagCoding(std::string_view name) : SampleCoding(name) {}

std::shared_ptr<MetadataAttribute> FlagCoding::GetFlag(std::string_view name) { return GetAttribute(name); }

std::vector<std::string> FlagCoding::GetFlagNames() { return GetAttributeNames(); }

std::shared_ptr<MetadataAttribute> FlagCoding::AddFlag(std::string_view name, int flag_mask,
                                                       std::string_view description) {
    return AddSamples(name, std::vector<int>{flag_mask}, description);
}

std::shared_ptr<MetadataAttribute> FlagCoding::AddFlag(std::string_view name, int flag_mask, int flag_value,
                                                       std::string_view description) {
    return AddSamples(name, std::vector<int>{flag_mask, flag_value}, description);
}

int FlagCoding::GetFlagMask(std::string_view name) {
    Guardian::AssertNotNull("name", name);
    std::shared_ptr<MetadataAttribute> attribute = GetAttribute(name);
    if (attribute == nullptr) {
        throw std::invalid_argument("flag '" + std::string(name) + "' not found");
    }
    //        Debug::AssertTrue(attribute->GetData()->IsInt());
    //        Debug::AssertTrue(attribute->GetData()->GetNumElems() == 1 || attribute->GetData()->GetNumElems() == 2);
    return attribute->GetData()->GetElemInt();
}

}  // namespace snapengine
}  // namespace alus
