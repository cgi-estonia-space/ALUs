#include "snap-engine-utilities/datamodel/metadata/abstract_metadata_i_o.h"

#include <iostream>
#include <iterator>
#include <stdexcept>

#include "snap-core/datamodel/metadata_attribute.h"
#include "snap-core/datamodel/metadata_element.h"
#include "snap-core/datamodel/product_data.h"

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
        std::cerr << e.what() << " " << name << " " << value << std::endl;
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
