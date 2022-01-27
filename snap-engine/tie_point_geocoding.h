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

#include "geocoding.h"
#include "raster_properties.h"
#include "tie_point_grid.h"

namespace alus::snapengine::geocoding {

class TiePointGeocoding : public Geocoding {
public:
    [[nodiscard]] Coordinates GetPixelCoordinates(PixelPosition pixel_position) const override;
    [[nodiscard]] Coordinates GetPixelCoordinates(PrecisePixelPosition pixel_position) const override;
    [[nodiscard]] Coordinates GetPixelCoordinates(std::tuple<double, double> pixel_position) const override;
    [[nodiscard]] Coordinates GetPixelCoordinates(double x, double y) const override;

    [[nodiscard]] PrecisePixelPosition GetPixelPosition(Coordinates pixel_coordinates) const override;
    [[nodiscard]] PrecisePixelPosition GetPixelPosition(std::tuple<double, double> pixel_coordinates) const override;
    [[nodiscard]] PrecisePixelPosition GetPixelPosition(double lon, double lat) const override;

    TiePointGeocoding(tiepointgrid::TiePointGrid latitude_grid, tiepointgrid::TiePointGrid longitude_grid)
        : latitude_grid_(latitude_grid), longitude_grid_(longitude_grid){};

    TiePointGeocoding(const TiePointGeocoding&) = delete;
    TiePointGeocoding& operator=(const TiePointGeocoding&) = delete;
    ~TiePointGeocoding() override = default;

    tiepointgrid::TiePointGrid latitude_grid_;
    tiepointgrid::TiePointGrid longitude_grid_;
};
}  // namespace alus::snapengine::geocoding