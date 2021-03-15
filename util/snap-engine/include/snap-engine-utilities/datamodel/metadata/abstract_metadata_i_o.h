/**
 * This file is a filtered duplicate of a SNAP's
 * org.esa.snap.engine_utilities.datamodel.metadata.AbstractMetadataIO.java ported for native code. Copied from a
 * snap-engine's(https://github.com/senbox-org/snap-engine) repository originally stated to be implemented by "Copyright
 * (C) 2016 by Array Systems Computing Inc. http://www.array.ca"
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
#pragma once

#include <memory>
#include <string>
#include <string_view>

#include "pugixml.hpp"

namespace alus::snapengine {
class MetadataElement;
class AbstractMetadataIO {
private:
    static void AddAttribute(const std::shared_ptr<MetadataElement>& meta, std::string_view name,
                             std::string_view value);
    static std::string LocalName(std::string_view namespaced_node_name);

public:
    /**
     * Add metadata from an XML file into the Metadata of the product
     *
     * @param xmlRoot      root element of xml file
     * @param metadataRoot MetadataElement to place it into
     */
    static void AddXMLMetadata(const pugi::xml_node& element, const std::shared_ptr<MetadataElement>& metadata_root);
};

}  // namespace alus::snapengine
