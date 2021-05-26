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

#include "raster_properties.h"

namespace alus {
namespace rasterutils {
inline __device__ __host__ alus::Coordinates CalculatePixelCoordinates(
    const alus::PrecisePixelPosition pixel_position, const alus::GeoTransformParameters geo_transform) {
    return {geo_transform.originLon + pixel_position.x * geo_transform.pixelSizeLon + geo_transform.pixelSizeLon / 2,
            geo_transform.originLat + pixel_position.y * geo_transform.pixelSizeLat + geo_transform.pixelSizeLat / 2};
}

inline __device__ __host__ alus::PrecisePixelPosition CalculatePixelPosition(const alus::Coordinates coordinates, const alus::GeoTransformParameters geo_transform) {
    return {
        (coordinates.lon - geo_transform.pixelSizeLon / 2 - geo_transform.originLon) / geo_transform.pixelSizeLon,
        (coordinates.lat - geo_transform.pixelSizeLat / 2 - geo_transform.originLat) / geo_transform.pixelSizeLat
    };
}

inline __device__ __host__ alus::PrecisePixelPosition TransformPixelPosition(const alus::PrecisePixelPosition origin_pixel_position, const alus::GeoTransformParameters origin_geo_transform, const alus::GeoTransformParameters target_geo_transform) {
    return CalculatePixelPosition(CalculatePixelCoordinates(origin_pixel_position, origin_geo_transform), target_geo_transform);
}
}  // namespace rasterutils
}  // namespace alus
