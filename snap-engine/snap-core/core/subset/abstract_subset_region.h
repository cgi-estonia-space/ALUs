/**
 * This file is a filtered duplicate of a SNAP's
 * org.esa.snap.core.subset.AbstractSubsetRegion.java
 * ported for native code.
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

#include "custom/rectangle.h"
#include "snap-core/core/datamodel/i_geo_coding.h"

namespace alus::snapengine {
class AbstractSubsetRegion {
protected:
    int border_pixels_;

    explicit AbstractSubsetRegion(int border_pixels);

    virtual void ValidateDefaultSize(int default_product_width, int default_product_height,
                                     std::string_view exception_message_prefix);

    AbstractSubsetRegion() = default;
    virtual ~AbstractSubsetRegion() = default;

public:
    AbstractSubsetRegion(const AbstractSubsetRegion&) = delete;
    AbstractSubsetRegion& operator=(const AbstractSubsetRegion&) = delete;

    virtual std::shared_ptr<custom::Rectangle> ComputeProductPixelRegion(
        std::shared_ptr<IGeoCoding> product_default_geo_coding, int default_product_width, int default_product_height,
        bool round_pixel_region) = 0;

    virtual std::shared_ptr<custom::Rectangle> ComputeBandPixelRegion(
        std::shared_ptr<IGeoCoding> product_default_geo_coding, std::shared_ptr<IGeoCoding> band_default_geo_coding,
        int default_product_width, int default_product_height, int default_band_width, int default_band_height,
        bool round_pixel_region) = 0;
};
}  // namespace alus::snapengine