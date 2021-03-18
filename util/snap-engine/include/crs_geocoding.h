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

#include "geocoding.h"

namespace alus {
namespace snapengine {
namespace geocoding {
class CrsGeocoding : public Geocoding {
   public:
    [[nodiscard]] Coordinates GetPixelCoordinates(PixelPosition pixel_position) const override;
    [[nodiscard]] Coordinates GetPixelCoordinates(PrecisePixelPosition pixel_position) const override;
    [[nodiscard]] Coordinates GetPixelCoordinates(std::tuple<double, double> pixel_position) const override;
    [[nodiscard]] Coordinates GetPixelCoordinates(double x, double y) const override;

    [[nodiscard]] PrecisePixelPosition GetPixelPosition(Coordinates pixel_coordinates) const override;
    [[nodiscard]] PrecisePixelPosition GetPixelPosition(std::tuple<double, double> pixel_coordinates) const override;
    [[nodiscard]] PrecisePixelPosition GetPixelPosition(double lon, double lat) const override;

    explicit CrsGeocoding(const GeoTransformParameters& geo_transform_parameters);
    CrsGeocoding(const CrsGeocoding&) = delete;
    CrsGeocoding& operator=(const CrsGeocoding&) = delete;
    ~CrsGeocoding() override = default;

    GeoTransformParameters geo_transform_parameters_;
};
}  // namespace geocoding
}  // namespace snapengine
}  // namespace alus