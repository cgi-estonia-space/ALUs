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

#include <algorithm>
#include <cctype>
#include <functional>
#include <ios>
#include <map>
#include <memory>
#include <optional>
#include <sstream>
#include <string_view>
#include <type_traits>
#include <vector>

#include <boost/lexical_cast.hpp>

#include "i_meta_data_reader.h"
#include "pugixml.hpp"
#include "snap-core/datamodel/tie_point_grid.h"
#include "spectral_band_info.h"

namespace alus {
namespace snapengine {
class MetadataElement;
class Product;
class PugixmlMetaDataReader : virtual public IMetaDataReader {
private:
    pugi::xml_document doc_;
    pugi::xml_parse_result result_;

    std::shared_ptr<MetadataElement> ImplToModel(std::string_view element_name);

    SpectralBandInfo GetSpectralBandInfo(pugi::xml_node& spectral_band);
    /**
     * Parses xml_node and reads information from requested top-level tag. This parse function does not search for
     * deep-nested values.
     *
     * @tparam T any type supported by boost::lexical_cast. Bool types support values "0", "1", "true", "false", and are
     * case-insensitive.
     * @param node pugi::xml_node whose child tag value should be returned.
     * @param child_tag Name of the top-level tag whose value should be parsed.
     * @return std::optional of the given type.
     */
    template <typename T>
    std::optional<T> ParseChildValue(const pugi::xml_node& node, std::string_view child_tag) {
        const std::string string_value = node.child_value(child_tag.data());
        if (string_value.empty()) {
            return std::nullopt;
        }
        return std::make_optional(boost::lexical_cast<T>(string_value));
    }

    std::unique_ptr<snapengine::TiePointGrid>  GetTiePointGrid(const pugi::xml_node& tie_point_grid_node);

public:
    PugixmlMetaDataReader() = default;
    void SetProduct(const std::shared_ptr<Product>& product) override;
    explicit PugixmlMetaDataReader(const std::shared_ptr<Product>& product);
    explicit PugixmlMetaDataReader(std::string_view file_name);
    [[nodiscard]] std::shared_ptr<MetadataElement> Read(std::string_view name) override;

    /**
     * Method for accessing Image_Interpretation tag of BEAM-DIMAP format and reading Bands info from the aforementioned
     * tag.
     *
     * @return Vector of SpectralBandInfo structs containing metadata information of the bands defined in the .dim file.
     * @todo Should be refactored and integrated into the SAFE or .dim file reader.
     */
    std::vector<SpectralBandInfo> ReadImageInterpretationTag();

    /**
     * Method for accessing Tie_Point_Grids tag of BEAM-DIMAP format and reading tie_point_grid info from the
     * aforementioned tag.
     *
     * @return Map of TiePointGrid names and their according TiePointGrid classes.
     * @todo Should be refactored and integrated into the SAFE or .dim file reader.
     */
    std::map<std::string, std::unique_ptr<snapengine::TiePointGrid>, std::less<>> ReadTiePointGridsTag();
    ~PugixmlMetaDataReader() override;
};

// Has to be outside of class namespace due to gcc not implementing C++ Core DR 727:
// https://gcc.gnu.org/bugzilla/show_bug.cgi?id=85282
template <>
inline std::optional<bool> PugixmlMetaDataReader::ParseChildValue(const pugi::xml_node& node, std::string_view child_tag) {
    std::string string_value = node.child_value(child_tag.data());
    if (string_value.empty()) {
        return std::nullopt;
    }
    try {
        return std::make_optional(boost::lexical_cast<bool>(string_value));
    } catch (const boost::bad_lexical_cast& e) {
        std::transform(string_value.begin(), string_value.end(), string_value.begin(), ::tolower);
        std::istringstream stream(string_value);
        bool parsed_boolean;
        stream >> std::boolalpha >> parsed_boolean;
        return std::make_optional(parsed_boolean);
    }
}

}  // namespace snapengine
}  // namespace alus
