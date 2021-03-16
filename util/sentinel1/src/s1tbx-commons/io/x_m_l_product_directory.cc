/**
 * This file is a filtered duplicate of a SNAP's
 * org.esa.s1tbx.commons.io.XMLProductDirectory.java
 * ported for native code.
 * Copied from (https://github.com/senbox-org/s1tbx). It was originally stated:
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
#include "s1tbx-commons/io/x_m_l_product_directory.h"

#include <fstream>
#include <iostream>

#include "snap-core/datamodel/metadata_element.h"
#include "snap-core/datamodel/product.h"
#include "snap-engine-utilities/datamodel/metadata/abstract_metadata.h"
#include "snap-engine-utilities/datamodel/metadata/abstract_metadata_i_o.h"

namespace alus::s1tbx {

std::shared_ptr<snapengine::MetadataElement> XMLProductDirectory::AddMetaData() {
    std::shared_ptr<snapengine::MetadataElement> root =
        std::make_shared<snapengine::MetadataElement>(snapengine::Product::METADATA_ROOT_NAME);

    auto root_element = xml_doc_.document_element();
    snapengine::AbstractMetadataIO::AddXMLMetadata(root_element,
                                                   snapengine::AbstractMetadata::AddOriginalProductMetadata(root));
    AddAbstractedMetadataHeader(root);

    return root;
}
XMLProductDirectory::XMLProductDirectory(const boost::filesystem::path& input_file)
    : AbstractProductDirectory(input_file) {}

void XMLProductDirectory::ReadProductDirectory() {
    std::fstream is;
    GetInputStream(GetRootFolder() + GetHeaderFileName(), is);
    if (is) {
        xml_doc_.load(is);
    }
    if (is.is_open()) {
        is.close();
    }
}

}  // namespace alus::s1tbx
