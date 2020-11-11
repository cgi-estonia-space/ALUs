/**
 * This file is a filtered duplicate of a SNAP's org.esa.snap.core.datamodel.AbstractBand.java
 * ported for native code.
 * Copied from a snap-engine's(https://github.com/senbox-org/snap-engine) repository originally stated
 * to be implemented by "Copyright (C) 2010 Brockmann Consult GmbH (info@brockmann-consult.de)"
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

#include <string_view>

#include "raster_data_node.h"

namespace alus {
namespace snapengine {

/**
 * The <code>AbstractBand</code> class provides a set of pixel access methods but does not provide an implementation of
 * the actual reading and writing of pixel data from or into a raster.
 *
 * original java version authors: Norman Fomferra, Sabine Embacher
 */
class AbstractBand : public RasterDataNode {
private:
    /**
     * The raster's width.
     */
    int raster_width_;

    /**
     * The raster's height.
     */
    int raster_height_;

public:
    AbstractBand(std::string_view name, int data_type, int raster_width, int raster_height)
        : RasterDataNode(name, data_type, static_cast<long>(raster_width) * static_cast<long>(raster_height)) {
        raster_width_ = raster_width;
        raster_height_ = raster_height;
    }

    /**
     * @return The width of the raster in pixels.
     */
    int GetRasterWidth() override { return raster_width_; }

    /**
     * @return The height of the raster in pixels.
     */
    int GetRasterHeight() override { return raster_height_; }
};
}  // namespace snapengine
}  // namespace alus
