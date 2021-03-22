/**
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
