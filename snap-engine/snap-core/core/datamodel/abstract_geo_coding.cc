/**
 * This file is a filtered duplicate of a SNAP's
 * org.esa.snap.core.datamodel.AbstractGeoCoding.java
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
#include "snap-core/core/datamodel/abstract_geo_coding.h"

namespace alus::snapengine {

MathTransformWKT AbstractGeoCoding::GetImageToMapTransform() {
    if (image_2_map_.empty()) {
        throw std::runtime_error("not yet ported/supported");
        // todo: port only if we use it somewhere
        //        synchronized (this) {
        //            if (image2Map == null) {
        //                try {
        //                    image2Map = CRS.findMathTransform(imageCRS, mapCRS);
        //                } catch (FactoryException e) {
        //                    throw new IllegalArgumentException(
        //                        "Not able to find a math transformation from image to map CRS.", e);
        //                }
        //            }
        //        }
    }
    return image_2_map_;
}
}  // namespace alus::snapengine