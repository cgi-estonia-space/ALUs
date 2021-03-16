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
#include <optional>
#include <string_view>
#include <vector>

#include <boost/lexical_cast.hpp>

#include "i_meta_data_reader.h"
#include "pugixml.hpp"
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
     * @tparam T any type supported by boost::lexical_cast
     * @param node pugi::xml_node whose child tag value should be returned.
     * @param child_tag Name of the top-level tag whose value should be parsed.
     * @return std::optional of the given type.
     *
     * @note Bool strings should be parsable boost::lexical_cast i.e. they should be either "1" or "0"
     */
    template <typename T>
    std::optional<T> ParseChildValue(pugi::xml_node node, std::string_view child_tag) {
        const std::string string_value = node.child_value(child_tag.data());
        if (string_value.empty()) {
            return std::nullopt;
        }
        auto parsed_value = std::make_optional(boost::lexical_cast<T>(string_value));
        return parsed_value;
    }

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
    ~PugixmlMetaDataReader() override;
};

}  // namespace snapengine
}  // namespace alus
