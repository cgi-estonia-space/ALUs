/**
 * This file is a filtered duplicate of a SNAP's
 * org.esa.s1tbx.io.sentinel1.Sentinel1Directory.java
 * ported for native code.
 * Copied from(https://github.com/senbox-org/s1tbx). It was originally stated:
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
#include <string_view>
#include <vector>

#include "snap-core/core/datamodel/band.h"
#include "snap-core/core/datamodel/metadata_attribute.h"
#include "snap-core/core/datamodel/metadata_element.h"

namespace alus::snapengine {
class Product;
}
namespace alus::s1tbx {

class BandInfo;
/**
 * Supports reading directories for level1, level2, and level0
 */
class ISentinel1Directory {
public:
    static constexpr std::string_view SENTINEL_DATE_FORMAT_PATTERN{"%Y-%m-%d %H:%M:%S"};

    virtual void Close() = 0;

    virtual void ReadProductDirectory() = 0;

    virtual std::shared_ptr<snapengine::Product> CreateProduct() = 0;

    virtual std::shared_ptr<BandInfo> GetBandInfo(const std::shared_ptr<snapengine::Band>& dest_band) = 0;

    virtual bool IsSLC() = 0;

    virtual std::shared_ptr<snapengine::MetadataElement> GetMetadataObject(
        const std::shared_ptr<snapengine::MetadataElement>& orig_prod_root, std::string_view metadata_object_name) {
        const std::shared_ptr<snapengine::MetadataElement>& metadata_section =
            orig_prod_root->GetElement("XFDU")->GetElement("metadataSection");
        const std::vector<std::shared_ptr<snapengine::MetadataElement>> metadata_objects =
            metadata_section->GetElements();

        for (auto elem : metadata_objects) {
            if (elem->GetAttribute("ID")->GetData()->GetElemString() == metadata_object_name) {
                return elem;
            }
        }
        return nullptr;
    }

    ISentinel1Directory() = default;
    ISentinel1Directory(const ISentinel1Directory&) = delete;
    ISentinel1Directory& operator=(const ISentinel1Directory&) = delete;
    virtual ~ISentinel1Directory() = default;
};
}  // namespace alus::s1tbx
