#pragma once

#include <vector>
#include <string_view>

#include "../../../external/pugixml/include/pugixml.hpp"

#include "metadata_element.h"
#include "i_meta_data_reader.h"

namespace alus {
namespace snapengine {

class PugixmlMetaDataReader : virtual public IMetaDataReader {
   protected:
    pugi::xml_document doc_;
    pugi::xml_parse_result result_;
    std::vector<MetadataElement> meta_data_latest_level_element_{};

   public:
    explicit PugixmlMetaDataReader(const std::string_view& file_name);
    // added this to be able to use different implementations
    [[nodiscard]] MetadataElement GetElement(std::string_view name) override;
    ~PugixmlMetaDataReader() override;
};

} //snapengine
}  // namespace alus
