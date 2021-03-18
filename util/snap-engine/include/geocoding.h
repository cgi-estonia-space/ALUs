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

#include <tuple>

#include "raster_properties.hpp"

namespace alus {
namespace snapengine {
namespace geocoding {
class Geocoding {
   public:
    [[nodiscard]] virtual Coordinates GetPixelCoordinates(PixelPosition pixel_position) const = 0;
    [[nodiscard]] virtual Coordinates GetPixelCoordinates(PrecisePixelPosition pixel_position) const = 0;
    [[nodiscard]] virtual Coordinates GetPixelCoordinates(std::tuple<double, double> pixel_position) const = 0;
    [[nodiscard]] virtual Coordinates GetPixelCoordinates(double x, double y) const = 0;

    [[nodiscard]] virtual PrecisePixelPosition GetPixelPosition(Coordinates pixel_coordinates) const = 0;
    [[nodiscard]] virtual PrecisePixelPosition GetPixelPosition(std::tuple<double, double> pixel_coordinates) const = 0;
    [[nodiscard]] virtual PrecisePixelPosition GetPixelPosition(double lon, double lat) const = 0;

    Geocoding() = default;
    Geocoding(const Geocoding&) = delete;
    Geocoding& operator=(const Geocoding&) = delete;
    virtual ~Geocoding()  = default;
};
}  // namespace geocoding
}  // namespace snapengine
}  // namespace alus