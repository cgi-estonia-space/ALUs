#pragma once

#include <string_view>
#include <vector>

#include "metadata_attribute.h"
#include "metadata_element.h"
#include "product_data.h"

namespace alus {
namespace snapengine {
class MetadataAttribute;
class MetadataElement;
class IMetaDataReader {
   protected:
//    const std::string_view file_name_;
    std::string_view file_name_;

   public:
    explicit IMetaDataReader(const std::string_view file_name) : file_name_(file_name){};
    //operate without implementation specific type
//    /*[[nodiscard]]*/ virtual MetadataElement GetElement(std::string_view name) const = 0;
    [[nodiscard]] virtual MetadataElement GetElement(std::string_view name) = 0;
//    [[nodiscard]] virtual std::vector<MetadataElement> GetElements() const = 0;
//    [[nodiscard]] virtual std::vector<MetadataAttribute> GetAttributes() const = 0;
//    [[nodiscard]] virtual MetadataElement GetParentElement() const = 0;
//    [[nodiscard]] virtual int GetNumElements() const = 0;
//    [[nodiscard]] virtual int GetNumAttributes() const = 0;
//    [[nodiscard]] virtual MetadataAttribute GetAttributeAt() const = 0;
//    [[nodiscard]] virtual std::string_view GetElementNames() const = 0;
//    [[nodiscard]] virtual std::string_view GetAttributeNames() const = 0;
//    [[nodiscard]] virtual bool ContainsElement(std::string_view name) const = 0;
//    [[nodiscard]] virtual bool ContainsAttribute(std::string_view name) const = 0;
//    [[nodiscard]] virtual int GetElementIndex(const MetadataElement &element) const = 0;
//    [[nodiscard]] virtual int GetAttributeIndex(const MetadataAttribute &attribute) const = 0;
//    [[nodiscard]] virtual MetadataAttribute GetAttribute(std::string_view attribute_name) const = 0;
//    [[nodiscard]] virtual int GetAttributeInt(std::string_view name, int default_value) const = 0;
//    [[nodiscard]] virtual int GetAttributeInt(std::string_view name) const = 0;
//    [[nodiscard]] virtual double GetAttributeDouble(std::string_view name, double default_value) const = 0;
//    [[nodiscard]] virtual double GetAttributeDouble(std::string_view name) const = 0;
//    [[nodiscard]] virtual snapengine::Utc GetAttributeUtc(std::string_view name, double default_value) const = 0;
//    [[nodiscard]] virtual snapengine::Utc GetAttributeUtc(std::string_view name) const = 0;
//    [[nodiscard]] virtual std::string_view GetAttributeString(std::string_view name, double default_value) const = 0;
//    [[nodiscard]] virtual std::string_view GetAttributeString(std::string_view name) const = 0;

    virtual ~IMetaDataReader() = default;
};
}  // namespace snapengine
}  // namespace alus