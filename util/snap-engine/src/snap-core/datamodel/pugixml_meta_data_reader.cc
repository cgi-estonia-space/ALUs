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
#include "pugixml_meta_data_reader.h"

#include <ios>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>

#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>

#include "dimap_product_constants.h"
#include "metadata_attribute.h"
#include "metadata_element.h"
#include "product.h"
#include "product_data.h"
#include "product_data_utc.h"
#include "snap-core/datamodel/tie_point_grid.h"
#include "spectral_band_info.h"

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
    //this->result_ = this->doc_.load_file(file_name.data(), pugi::parse_default | pugi::parse_declaration);
    std::cout << "Reading from " << file_name_ << std::endl;
    this->result_ = this->doc_.load_file(file_name_.data(), pugi::parse_default | pugi::parse_declaration);
    if (!this->result_) {
        // todo: add exception handling wrapper or handle directly (use PugixmlErrorException)
        throw std::runtime_error("unable to load file " + file_name);
    }
    return ImplToModel(element_name);
}

SpectralBandInfo PugixmlMetaDataReader::GetSpectralBandInfo(pugi::xml_node& spectral_band) {
    const auto band_index = std::stoi(spectral_band.child_value(DimapProductConstants::TAG_BAND_INDEX.data()));
    const std::string band_name = spectral_band.child_value(DimapProductConstants::TAG_BAND_NAME.data());
    const auto product_data_type =
        ProductData::GetType(spectral_band.child_value(DimapProductConstants::TAG_DATA_TYPE.data()));
    bool log_10_scaled;
    std::istringstream(spectral_band.child_value(DimapProductConstants::TAG_SCALING_LOG_10.data())) >> std::boolalpha >>
        log_10_scaled;
    bool no_data_value_used;
    std::istringstream(spectral_band.child_value(DimapProductConstants::TAG_NO_DATA_VALUE_USED.data())) >>
        std::boolalpha >> no_data_value_used;

    // Parse optional values
    const auto band_description =
        ParseChildValue<std::string>(spectral_band, DimapProductConstants::TAG_BAND_DESCRIPTION.data());
    const auto physical_unit =
        ParseChildValue<std::string>(spectral_band, DimapProductConstants::TAG_PHYSICAL_UNIT.data());
    const auto solar_flux = ParseChildValue<double>(spectral_band, DimapProductConstants::TAG_SOLAR_FLUX.data());
    const auto spectral_band_index =
        ParseChildValue<int>(spectral_band, DimapProductConstants::TAG_SPECTRAL_BAND_INDEX);
    const auto band_wavelength = ParseChildValue<double>(spectral_band, DimapProductConstants::TAG_BAND_WAVELEN.data());
    const auto bandwidth = ParseChildValue<double>(spectral_band, DimapProductConstants::TAG_BANDWIDTH.data());
    const auto scaling_factor =
        ParseChildValue<double>(spectral_band, DimapProductConstants::TAG_SCALING_FACTOR.data());
    const auto scaling_offset =
        ParseChildValue<double>(spectral_band, DimapProductConstants::TAG_SCALING_OFFSET.data());
    const auto no_data_value = ParseChildValue<double>(spectral_band, DimapProductConstants::TAG_NO_DATA_VALUE.data());
    const auto valid_mask_term =
        ParseChildValue<std::string>(spectral_band, DimapProductConstants::TAG_VALID_MASK_TERM.data());

    return {band_index,       band_name,      product_data_type, log_10_scaled,       no_data_value_used,
            band_description, physical_unit,  solar_flux,        spectral_band_index, band_wavelength,
            bandwidth,        scaling_factor, scaling_offset,    no_data_value,       valid_mask_term};
}

std::unique_ptr<snapengine::TiePointGrid> PugixmlMetaDataReader::GetTiePointGrid(const pugi::xml_node& tie_point_grid_node) {
    const std::string name = tie_point_grid_node.child_value(DimapProductConstants::TAG_TIE_POINT_GRID_NAME.data());

    const auto grid_width =
        std::stoi(tie_point_grid_node.child_value(DimapProductConstants::TAG_TIE_POINT_NCOLS.data()));
    const auto grid_height =
        std::stoi(tie_point_grid_node.child_value(DimapProductConstants::TAG_TIE_POINT_NROWS.data()));
    const auto offset_x =
        std::stod(tie_point_grid_node.child_value(DimapProductConstants::TAG_TIE_POINT_OFFSET_X.data()));
    const auto offset_y =
        std::stod(tie_point_grid_node.child_value(DimapProductConstants::TAG_TIE_POINT_OFFSET_Y.data()));
    const auto subsampling_x =
        std::stod(tie_point_grid_node.child_value(DimapProductConstants::TAG_TIE_POINT_STEP_X.data()));
    const auto subsampling_y =
        std::stod(tie_point_grid_node.child_value(DimapProductConstants::TAG_TIE_POINT_STEP_Y.data()));

    auto is_cyclic = ParseChildValue<bool>(tie_point_grid_node, DimapProductConstants::TAG_TIE_POINT_CYCLIC);

    std::vector<float> tie_points(grid_width * grid_height);
    return std::make_unique<TiePointGrid>(
        name, grid_width, grid_height, offset_x, offset_y, subsampling_x, subsampling_y, tie_points,
        (is_cyclic && is_cyclic.value() ? TiePointGrid::DISCONT_AT_180 : TiePointGrid::DISCONT_NONE));
}

std::map<std::string, std::unique_ptr<snapengine::TiePointGrid>, std::less<>> PugixmlMetaDataReader::ReadTiePointGridsTag() {
    std::map<std::string, std::unique_ptr<snapengine::TiePointGrid>, std::less<>> tie_point_grids;

    std::string file_name;

    if (product_) {
        file_name = product_->GetFileLocation().parent_path().generic_path().string() +
                    boost::filesystem::path::preferred_separator + product_->GetName() + ".xml";
    } else if (!file_name_.empty()) {
        file_name = file_name_;
    } else {
        throw std::runtime_error("no source file for metadata provided");
    }

    result_ = doc_.load_file(file_name.data(), pugi::parse_default | pugi::parse_declaration);
    if (!result_) {
        throw std::runtime_error("unable to load file " + file_name);
    }

    const auto root = doc_.document_element();
    const auto tie_point_grids_node = root.select_node(DimapProductConstants::TAG_TIE_POINT_GRIDS.data());
    if (!tie_point_grids_node) {
        throw std::runtime_error(file_name + " does not contain " + DimapProductConstants::TAG_TIE_POINT_GRIDS.data() +
                                 " tag");
    }

    const auto tie_point_grid_nodes = tie_point_grids_node.node().children();
    for (const auto& tie_point_grid_node : tie_point_grid_nodes) {
        if (tie_point_grid_node.name() == DimapProductConstants::TAG_TIE_POINT_NUM_TIE_POINT_GRIDS) {
            continue;
        }
        auto tie_point_grid = GetTiePointGrid(tie_point_grid_node);
        tie_point_grids.try_emplace(tie_point_grid->GetName(), std::move(tie_point_grid));
    }

    return tie_point_grids;
}

std::vector<SpectralBandInfo> PugixmlMetaDataReader::ReadImageInterpretationTag() {
    std::vector<SpectralBandInfo> bands_info;

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

    const auto root = doc_.document_element();
    const auto image_interpretation_node = root.select_node(DimapProductConstants::TAG_IMAGE_INTERPRETATION.data());
    if (!image_interpretation_node) {
        throw std::runtime_error(file_name + " does not contain " +
                                 DimapProductConstants::TAG_IMAGE_INTERPRETATION.data() + " tag");
    }

    const auto spectral_bands = image_interpretation_node.node().children();
    for (auto&& spectral_band : spectral_bands) {
        bands_info.push_back(GetSpectralBandInfo(spectral_band));
    }

    return bands_info;
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
