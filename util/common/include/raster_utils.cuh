#pragma once

#include "raster_properties.hpp"

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
