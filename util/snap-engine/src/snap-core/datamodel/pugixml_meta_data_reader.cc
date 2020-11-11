#include "pugixml_meta_data_reader.h"

#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>

#include "dimap_product_constants.h"
#include "metadata_attribute.h"
#include "metadata_element.h"
#include "product.h"
#include "product_data.h"
#include "product_data_utc.h"

namespace alus {
namespace snapengine {

PugixmlMetaDataReader::PugixmlMetaDataReader(const std::shared_ptr<Product>& product) : IMetaDataReader(product) {}
PugixmlMetaDataReader::PugixmlMetaDataReader(const std::string_view file_name) : IMetaDataReader(file_name) {}

std::shared_ptr<MetadataElement> PugixmlMetaDataReader::Read(std::string_view element_name) {
    std::string file_name;
    if (product_) {
        file_name = product_->GetFileLocation().parent_path().generic_path().string() +
                    boost::filesystem::path::preferred_separator + product_->GetName() + ".xml";
    } else if (!file_name_.empty()) {
        file_name = file_name_;
    } else {
        throw std::runtime_error("no source file for metadata provided");
    }
    this->result_ = this->doc_.load_file(file_name.data(), pugi::parse_default | pugi::parse_declaration);
    if (!this->result_) {
        // todo: add exception handling wrapper or handle directly (use PugixmlErrorException)
        throw std::runtime_error("unable to load file " + file_name);
    }
    return ImplToModel(element_name);
}

std::shared_ptr<MetadataElement> PugixmlMetaDataReader::ImplToModel(std::string_view element_name) {
    std::shared_ptr<MetadataElement> start_element = std::make_shared<MetadataElement>(element_name);

    struct SimpleWalker : pugi::xml_tree_walker {
        // simple access to latest element on each depth level
        std::vector<std::shared_ptr<MetadataElement>> meta_data_latest_level_element_{};

        explicit SimpleWalker(const std::shared_ptr<MetadataElement>& e) {
            meta_data_latest_level_element_.push_back(e);
        }

        bool for_each(pugi::xml_node& node) override {
            if (node.name() == DimapProductConstants::TAG_METADATA_ELEMENT) {
                for (pugi::xml_attribute_iterator ait = node.attributes_begin(); ait != node.attributes_end(); ++ait) {
                    if (ait->name() == std::string_view{"name"} && ait->value() != std::string_view{""}) {
                        auto current = std::make_shared<MetadataElement>(ait->value());
                        // build local tree structure of elements
                        meta_data_latest_level_element_[depth()]->AddElement(current);
                        // update latest references for each level
                        if (static_cast<int>(meta_data_latest_level_element_.size()) < (depth() + 2)) {
                            // if something has never been added to current depth level add it to the end of vector
                            // which holds references
                            meta_data_latest_level_element_.push_back(current);
                        } else {
                            // if already something then just replace latest reference on current depth level
                            meta_data_latest_level_element_[depth() + 1] = current;
                        }
                    }
                }
            } else if (node.name() == DimapProductConstants::TAG_METADATA_ATTRIBUTE && !node.text().empty()) {
                std::string_view att_name;
                std::string_view att_type;
                //		std::string_view att_mode;
                std::optional<std::string> att_desc;
                std::optional<std::string> att_unit;
                bool read_only = true;
                std::string_view att_value{node.text().as_string()};
                for (pugi::xml_attribute_iterator ait = node.attributes_begin(); ait != node.attributes_end(); ++ait) {
                    if (ait->name() == std::string_view{DimapProductConstants::ATTRIB_NAME} &&
                        ait->value() != std::string_view{""}) {
                        att_name = ait->value();
                    } else if (ait->name() == std::string_view{DimapProductConstants::ATTRIB_TYPE} &&
                               ait->value() != std::string_view{""}) {
                        att_type = ait->value();
                    } else if (ait->name() == std::string_view{DimapProductConstants::ATTRIB_MODE}) {
                        read_only = !boost::iequals("rw", ait->value());
                    } else if (ait->name() == std::string_view{DimapProductConstants::ATTRIB_DESCRIPTION}) {
                        att_desc = ait->value();
                    } else if (ait->name() == std::string_view{DimapProductConstants::ATTRIB_UNIT}) {
                        att_unit = ait->value();
                    } else {
                        continue;
                    }
                }

                auto type = ProductData::GetType(att_type);
                std::shared_ptr<ProductData> data = nullptr;
                if (type == ProductData::TYPE_ASCII) {
                    // todo  att_value to vector<int8_t> maybe move this transformer.
                    data = ProductData::CreateInstance(att_value);
                } else if (type == ProductData::TYPE_UTC) {
                    if (att_value.find(",") != std::string::npos) {
                        // *************************************************
                        // this case is necessary for backward compatibility
                        // *************************************************
                        std::vector<std::string> data_values;
                        boost::split(data_values, att_value, boost::is_any_of(","));
                        data = ProductData::CreateInstance(type);
                        data->SetElems(data_values);
                    } else {
                        std::shared_ptr<Utc> utc;
                        try {
                            utc = Utc::Parse(att_value);
                        } catch (const std::exception& e) {
                            std::cerr << e.what();
                        }
                        data = utc;
                    }
                } else {
                    std::vector<std::string> data_values;
                    boost::split(data_values, att_value, boost::is_any_of(","));
                    auto length = data_values.size();
                    data = ProductData::CreateInstance(type, length);
                    data->SetElems(data_values);
                }

                if (data) {
                    auto current = std::make_shared<MetadataAttribute>(att_name, data, read_only);
                    current->SetDescription(att_desc);
                    current->SetUnit(att_unit);
                    // build local tree structure of elements
                    meta_data_latest_level_element_[depth()]->AddAttribute(current);
                }
            } else {
                // throw exception?
            }
            return true;
        }
    };

    // use implementations specific to construct metadata element object
    pugi::xpath_variable_set vars;
    vars.add("name", pugi::xpath_type_string);
    std::string query{"//"};
    query.append(DimapProductConstants::TAG_METADATA_ELEMENT).append("[@name = string($name)]");
    pugi::xpath_query name_query(query.data(), &vars);
    vars.set("name", element_name.data());
    pugi::xpath_node_set query_result = name_query.evaluate_node_set(doc_);
    // translate from xpath_node_set to MetaDataElement
    SimpleWalker walker(start_element);
    query_result.first().node().traverse(walker);

    return start_element;
}

void PugixmlMetaDataReader::SetProduct(const std::shared_ptr<Product>& product) { product_ = product; }

PugixmlMetaDataReader::~PugixmlMetaDataReader() = default;
}  // namespace snapengine
}  // namespace alus
