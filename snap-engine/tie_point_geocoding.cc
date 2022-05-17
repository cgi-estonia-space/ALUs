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
#include "tie_point_geocoding.h"
#include <stdexcept>

namespace alus::snapengine::geocoding {
Coordinates TiePointGeocoding::GetPixelCoordinates(alus::PixelPosition pixel_position) const {
    return this->GetPixelCoordinates(pixel_position.x + 0.5, pixel_position.y + 0.5);
}
Coordinates TiePointGeocoding::GetPixelCoordinates(std::tuple<double, double> pixel_position) const {
    return this->GetPixelCoordinates(std::get<0>(pixel_position), std::get<1>(pixel_position));
}
Coordinates TiePointGeocoding::GetPixelCoordinates(double x, double y) const {
    double latitude = tiepointgrid::GetPixelDouble(x, y, &this->latitude_grid_);
    double longitude = tiepointgrid::GetPixelDouble(x, y, &this->longitude_grid_);
    return {longitude, latitude};
}

Coordinates TiePointGeocoding::GetPixelCoordinates(PrecisePixelPosition pixel_position) const {
    return this->GetPixelCoordinates(pixel_position.x, pixel_position.y);
}

PrecisePixelPosition TiePointGeocoding::GetPixelPosition([[maybe_unused]] alus::Coordinates pixel_coordinates) const {
    throw std::runtime_error("This function is not implemented yet.");
}

PrecisePixelPosition TiePointGeocoding::GetPixelPosition(
    [[maybe_unused]] std::tuple<double, double> pixel_coordinates) const {
    throw std::runtime_error("This function is not implemented yet.");
}

PrecisePixelPosition TiePointGeocoding::GetPixelPosition([[maybe_unused]] double lon,
                                                         [[maybe_unused]] double lat) const {
    throw std::runtime_error("This function is not implemented yet.");
}
}  // namespace alus::snapengine::geocoding