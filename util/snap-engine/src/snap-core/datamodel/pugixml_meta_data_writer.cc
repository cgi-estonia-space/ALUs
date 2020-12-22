#include "pugixml_meta_data_writer.h"

#include <string>

#include "dimap_product_constants.h"
#include "metadata_element.h"
#include "product.h"

namespace alus {
namespace snapengine {

void PugixmlMetaDataWriter::Write() {
    // todo: copy might be fast if is_modified = false for everything
    // write internal model to xml file using pugixml
    ModelToImpl();
    doc_.save_file(std::string(product_->GetFileLocation().parent_path().generic_path().string() +
                               boost::filesystem::path::preferred_separator + product_->GetName() + ".xml")
                       .c_str());
}

void PugixmlMetaDataWriter::ModelAttributesToPugixmlNodes(const std::shared_ptr<MetadataElement>& elem,
                                                          pugi::xml_node& node) {
    if (!elem->GetAttributes().empty()) {
        for (const auto& md_atr : elem->GetAttributes()) {
            pugi::xml_node child =
                node.append_child(std::string(DimapProductConstants::TAG_METADATA_ATTRIBUTE).c_str());
            if (!md_atr->GetName().empty()) {
                child.append_attribute(std::string(DimapProductConstants::ATTRIB_NAME).c_str()) =
                    md_atr->GetName().c_str();
            }
            if (md_atr->GetDescription()) {
                child.append_attribute(std::string(DimapProductConstants::ATTRIB_DESCRIPTION).c_str()) =
                    md_atr->GetDescription().value().c_str();
            }
            if (md_atr->GetUnit()) {
                child.append_attribute(std::string(DimapProductConstants::ATTRIB_UNIT).c_str()) =
                    md_atr->GetUnit().value().c_str();
            }
            auto data_type_string = md_atr->GetData()->GetTypeString();
            if (!data_type_string.empty()) {
                child.append_attribute(std::string(DimapProductConstants::ATTRIB_TYPE).c_str()) =
                    data_type_string.c_str();
            }
            if (!md_atr->IsReadOnly()) {
                child.append_attribute(std::string(DimapProductConstants::ATTRIB_MODE).c_str()) = "rw";
            }
            if (md_atr->GetNumDataElems() > 1 && (!(ProductData::TYPESTRING_ASCII == data_type_string)) &&
                (!(ProductData::TYPESTRING_UTC == data_type_string))) {
                child.append_attribute(std::string(DimapProductConstants::ATTRIB_ELEMS).c_str()) =
                    std::to_string(md_atr->GetNumDataElems()).c_str();
            }
            auto nodechild = child.append_child(pugi::node_pcdata);
            nodechild.set_value(md_atr->GetData()->GetElemString().c_str());
        }
    }
}

void PugixmlMetaDataWriter::ModelElementsToPugixmlNodes(const std::shared_ptr<MetadataElement>& parent_elem,
                                                        pugi::xml_node& node) {
    for (const auto& child_elem : parent_elem->GetElements()) {
        // create root node
        auto child = node.append_child(std::string(DimapProductConstants::TAG_METADATA_ELEMENT).c_str());
        // add attribute name and value of name if they exist
        if (!child_elem->GetName().empty()) {
            child.append_attribute(std::string(DimapProductConstants::ATTRIB_NAME).c_str()) =
                child_elem->GetName().c_str();
        }
        ModelAttributesToPugixmlNodes(child_elem, child);
        ModelElementsToPugixmlNodes(child_elem, child);
    }
}

void PugixmlMetaDataWriter::ModelToImpl() {
    // now get root node
    pugi::xml_node node = doc_.append_child(std::string(DimapProductConstants::TAG_METADATA_ELEMENT).c_str());
    if (!product_->GetMetadataRoot()->GetName().empty()) {
        node.append_attribute(std::string(DimapProductConstants::ATTRIB_NAME).c_str()) =
            product_->GetMetadataRoot()->GetName().c_str();
    }
    ModelElementsToPugixmlNodes(product_->GetMetadataRoot(), node);
}

void PugixmlMetaDataWriter::SetProduct(const std::shared_ptr<Product>& product) { product_ = product; }

PugixmlMetaDataWriter::PugixmlMetaDataWriter(const std::shared_ptr<Product>& product) : IMetaDataWriter(product) {}

}  // namespace snapengine
}  // namespace alus
