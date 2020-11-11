#pragma once

#include <memory>

#include "i_meta_data_writer.h"
#include "pugixml.hpp"

namespace alus {
namespace snapengine {
class MetadataElement;
class Product;
/**
 * probably temporary class which provides means to write internal metadata model to xml documents using pugixml library
 *
 * simply put: interal data model -> output_metadata_file.xml
 */
class PugixmlMetaDataWriter : virtual public IMetaDataWriter {
private:
    pugi::xml_document doc_;
    // internal model to pugi::xml_document
    void ModelToImpl();
    void ModelElementsToPugixmlNodes(const std::shared_ptr<MetadataElement>& parent_elem, pugi::xml_node& node);
    static void ModelAttributesToPugixmlNodes(const std::shared_ptr<MetadataElement>& elem, pugi::xml_node& node);

public:
    PugixmlMetaDataWriter() = default;
    explicit PugixmlMetaDataWriter(const std::shared_ptr<Product>& product);
    void Write() override;
    void SetProduct(const std::shared_ptr<Product>& product) override;
    ~PugixmlMetaDataWriter() override = default;
};

}  // namespace snapengine
}  // namespace alus
