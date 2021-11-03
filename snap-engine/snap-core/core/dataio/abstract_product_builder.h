/**
 * This file is a filtered duplicate of a SNAP's
 * org.esa.snap.core.dataio.AbstractProductBuilder.java
 * ported for native code.
 * Copied from (https://github.com/senbox-org/snap-engine). It was originally stated:
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

#include <map>
#include <memory>
#include <string>

#include "abstract_product_reader.h"
#include "snap-core/core/datamodel/band.h"
#include "snap-core/core/datamodel/product.h"
#include "snap-core/core/datamodel/raster_data_node.h"

namespace alus::snapengine {

class AbstractProductBuilder : public AbstractProductReader {
public:
    AbstractProductBuilder(bool source_product_owner);

    std::shared_ptr<Product>& GetSourceProduct() { return source_product_; }
    int GetSceneRasterWidth() { return scene_raster_width_; }

    int GetSceneRasterHeight() { return scene_raster_height_; }

protected:
    bool source_product_owner_;
    std::shared_ptr<Product> source_product_;
    int scene_raster_width_;
    int scene_raster_height_;
    std::string new_product_name_;
    std::string new_product_desc_;
    std::map<std::shared_ptr<Band>, std::shared_ptr<RasterDataNode>> band_map_;

private:
};

}  // namespace alus::snapengine