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
#include "crs_geocoding.h"

namespace alus::snapengine::geocoding {
Coordinates CrsGeocoding::GetPixelCoordinates(double x, double y) const {
    auto geo_transform = this->geo_transform_parameters_;
    double longitude = geo_transform.originLon + geo_transform.pixelSizeLon * (x) + geo_transform.pixelSizeLon / 2;
    double latitude = geo_transform.originLat + geo_transform.pixelSizeLat * (y) + geo_transform.pixelSizeLat / 2;

    return {longitude, latitude};
}
Coordinates CrsGeocoding::GetPixelCoordinates(std::tuple<double, double> pixel_position) const {
    return GetPixelCoordinates(std::get<0>(pixel_position), std::get<1>(pixel_position));
}
alus::Coordinates CrsGeocoding::GetPixelCoordinates(PixelPosition pixel_position) const {
    return GetPixelCoordinates(pixel_position.x, pixel_position.y);
}

Coordinates CrsGeocoding::GetPixelCoordinates(PrecisePixelPosition pixel_position) const {
    return GetPixelCoordinates(pixel_position.x, pixel_position.y);
}

CrsGeocoding::CrsGeocoding(const GeoTransformParameters& geo_transform_parameters)
    : geo_transform_parameters_(geo_transform_parameters) {}

PrecisePixelPosition CrsGeocoding::GetPixelPosition(Coordinates pixel_coordinates) const {
    return GetPixelPosition(pixel_coordinates.lon, pixel_coordinates.lat);
}

PrecisePixelPosition CrsGeocoding::GetPixelPosition(std::tuple<double, double> pixel_coordinates) const {
    return GetPixelPosition(std::get<0>(pixel_coordinates), std::get<1>(pixel_coordinates));
}

PrecisePixelPosition CrsGeocoding::GetPixelPosition(double lon, double lat) const {
    auto geo_transform = this->geo_transform_parameters_;
    double x = (lon - geo_transform.pixelSizeLon / 2 - geo_transform.originLon) / geo_transform.pixelSizeLon;
    double y = (lat - geo_transform.pixelSizeLat / 2 - geo_transform.originLat) / geo_transform.pixelSizeLat;

    return {x, y};
}
}  // namespace alus::snapengine::geocoding