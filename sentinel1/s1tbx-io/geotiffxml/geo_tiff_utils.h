/**
 * This file is a filtered duplicate of a SNAP's
 * org.esa.s1tbx.io.geotiffxml.GeoTiffUtils.java
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

#include "custom/i_image_reader.h"

namespace alus::s1tbx {

class GeoTiffUtils {
public:
    //    todo: looks like inputstream will be given to concrete implementation of image reader which can decode given
    //    inputstream static IImageReader GetTiffIIOReader(const std::istream& stream);
    // TEMPORARY PLACEHOLDER SOLUTION TO PROVIDE GDAL (or some future alternative)
    static std::shared_ptr<snapengine::custom::IImageReader> GetTiffIIOReader();
};

}  // namespace alus::s1tbx
