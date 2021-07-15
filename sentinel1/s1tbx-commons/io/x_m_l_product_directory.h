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
#pragma once

#include "s1tbx-commons/io/abstract_product_directory.h"

#include <boost/filesystem/path.hpp>

#include "pugixml.hpp"

namespace alus::snapengine {
class MetadataElement;
}

namespace alus::s1tbx {

/**
 * This class represents a product directory.
 */
class XMLProductDirectory : public AbstractProductDirectory {
protected:
    pugi::xml_document xml_doc_;
    explicit XMLProductDirectory(const boost::filesystem::path& input_file);
    std::shared_ptr<snapengine::MetadataElement> AddMetaData() override;

public:
    void ReadProductDirectory() override;
};

}  // namespace alus::s1tbx
