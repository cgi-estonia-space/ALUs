/**
 * This file is a filtered duplicate of a SNAP's
 * org.esa.snap.engine_utilities.datamodel.metadata.AbstractMetadataIO.java
 * ported for native code.
 * Copied from(https://github.com/senbox-org/snap-engine). It was originally stated:
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
#include "snap-engine-utilities/engine-utilities/datamodel/metadata/abstract_metadata_i_o.h"

#include <iterator>
#include <stdexcept>

#include "alus_log.h"
#include "snap-core/core/datamodel/metadata_attribute.h"
#include "snap-core/core/datamodel/metadata_element.h"
#include "snap-core/core/datamodel/product_data.h"

namespace alus::snapengine {

void AbstractMetadataIO::AddXMLMetadata(const pugi::xml_node& xml_root,
                                        const std::shared_ptr<MetadataElement>& metadata_root) {
    std::string root_name = LocalName(xml_root.name());
    size_t child_nodes = std::distance(xml_root.children().begin(), xml_root.children().end());
    size_t child_attributes = std::distance(xml_root.attributes_begin(), xml_root.attributes_end());
    if (child_nodes == 1 && child_attributes == 0 &&
        xml_root.first_child().type() == pugi::xml_node_type::node_pcdata) {
        if (xml_root.first_child().value()) {
            AddAttribute(metadata_root, root_name, xml_root.first_child().value());
        }
    } else if (child_nodes == 1 && xml_root.first_child().type() == pugi::xml_node_type::node_pcdata) {
        auto meta_elem = std::make_shared<MetadataElement>(root_name);
        if (xml_root.first_child().value()) {
            AddAttribute(meta_elem, root_name, xml_root.first_child().value());
        }
        auto xml_attribs = xml_root.attributes();
        for (auto const& a_child : xml_attribs) {
            AddAttribute(meta_elem, LocalName(a_child.name()), a_child.value());
        }
        metadata_root->AddElement(meta_elem);
    } else {
        std::shared_ptr<MetadataElement> meta_elem = std::make_shared<MetadataElement>(root_name);
        auto children = xml_root.children();
        for (auto const& a_child : children) {
            AddXMLMetadata(a_child, meta_elem);
        }
        auto xml_attribs = xml_root.attributes();
        for (auto const& a_child : xml_attribs) {
            AddAttribute(meta_elem, LocalName(a_child.name()), a_child.value());
        }
        metadata_root->AddElement(meta_elem);
    }
}

void AbstractMetadataIO::AddAttribute(const std::shared_ptr<MetadataElement>& meta, std::string_view name,
                                      std::string_view value) {
    try {
        auto attribute = std::make_shared<MetadataAttribute>(name, ProductData::TYPE_ASCII, 1);
        if (value.empty()) {
            value = " ";
        }
        attribute->GetData()->SetElems(value);
        meta->AddAttribute(attribute);
    } catch (const std::exception& e) {
        LOGW << e.what() << " " << name << " " << value;
    }
}

std::string AbstractMetadataIO::LocalName(std::string_view namespaced_node_name) {
    if (namespaced_node_name.find_last_of(':') != std::string::npos &&
        namespaced_node_name.find_last_of(':') + 1 != std::string::npos) {
        return std::string(namespaced_node_name.substr(namespaced_node_name.find_last_of(':') + 1));
    }
    return std::string(namespaced_node_name);
}

}  // namespace alus::snapengine
