/**
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

#include "pos_vector.h"

namespace alus {
namespace s1tbx {

/**
 * A copy of RangeDopplerGeoCodingOp.java's private PositionData class.
 * BackGeocodingOp.java has the same private class.
 */
struct PositionData final {
    snapengine::PosVector earth_point;
    snapengine::PosVector sensor_pos;
    double azimuth_index;
    double range_index;
    double slant_range;
};

}  // namespace s1tbx
}  // namespace slap