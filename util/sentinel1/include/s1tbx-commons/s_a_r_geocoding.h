/**
 * This file is a filtered duplicate of a SNAP's
 * org.esa.s1tbx.commons.SARGeocoding.java
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

#include "snap-core/datamodel/tie_point_grid.h"

namespace alus::s1tbx {

/**
 * SAR specific common functions
 */
class SARGeocoding {
public:
    static bool IsNearRangeOnLeft(const std::shared_ptr<snapengine::TiePointGrid>& incidence_angle,
                                  int source_image_width) {
        // for products without incidence angle tpg just assume left facing
        if (incidence_angle == nullptr) {
            return true;
        }
        double incidence_angle_to_first_pixel = incidence_angle->GetPixelDouble(0, 0);
        double incidence_angle_to_last_pixel = incidence_angle->GetPixelDouble(source_image_width - 1, 0);
        return (incidence_angle_to_first_pixel <= incidence_angle_to_last_pixel);
    }
};

}  // namespace alus::s1tbx
