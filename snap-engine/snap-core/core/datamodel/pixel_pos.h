/**
 * This file is a filtered duplicate of a SNAP's
 * org.esa.snap.core.datamodel.PixelPos.java
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

#include <cmath>

namespace alus::snapengine {
class PixelPos {
    /**
     * Constructs and initializes a <code>PixelPos</code> with coordinate (0,&nbsp;0).
     */
public:
    double x_;
    double y_;
    PixelPos() = default;

    /**
     * Constructs and initializes a <code>PixelPos</code> with the specified coordinate.
     *
     * @param x the x component of the coordinate
     * @param y the y component of the coordinate
     */
    PixelPos(double x, double y) : x_(x), y_(y){};

    /**
     * Tests whether or not this pixel position is valid.
     *
     * @return true, if so
     */
    [[nodiscard]] bool IsValid() const { return !(std::isnan(x_) || std::isnan(y_)); }

    /**
     * Sets this pixel position so that is becomes invalid.
     */
    void SetInvalid() {
        x_ = std::nan("");
        y_ = std::nan("");
    }
};
}  // namespace alus::snapengine
