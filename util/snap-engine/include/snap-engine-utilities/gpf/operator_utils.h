/**
 * This file is a filtered duplicate of a SNAP's
 * org.esa.snap.engine_utilities.gpf.OperatorUtils.java
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

#include <string>
#include <string_view>

namespace alus {
namespace snapengine {

/**
 * Helper methods for working with Operators
 */
class OperatorUtils {
private:
public:
    static constexpr std::string_view TPG_SLANT_RANGE_TIME = "slant_range_time";
    static constexpr std::string_view TPG_INCIDENT_ANGLE = "incident_angle";
    static constexpr std::string_view TPG_ELEVATION_ANGLE = "elevation_angle";
    static constexpr std::string_view TPG_LATITUDE = "latitude";
    static constexpr std::string_view TPG_LONGITUDE = "longitude";

    static std::string GetPolarizationFromBandName(std::string_view band_name);
};

}  // namespace snapengine
}  // namespace alus
