#pragma once

#include <string_view>
#include <memory>

#include "i_meta_data_reader.h"
#include "pugixml.hpp"

namespace alus {
namespace snapengine {
class MetadataElement;
class Product;
class PugixmlMetaDataReader : virtual public IMetaDataReader {
private:
    pugi::xml_document doc_;
    pugi::xml_parse_result result_;

    std::shared_ptr<MetadataElement> ImplToModel(std::string_view element_name);

public:
    PugixmlMetaDataReader() = default;
    void SetProduct(const std::shared_ptr<Product>& product) override;
    explicit PugixmlMetaDataReader(const std::shared_ptr<Product>& product);
    explicit PugixmlMetaDataReader(std::string_view file_name);
    [[nodiscard]] std::shared_ptr<MetadataElement> Read(std::string_view name) override;
    ~PugixmlMetaDataReader() override;
};

}  // namespace snapengine
}  // namespace alus
