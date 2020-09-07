#include "crs_geocoding.cuh"

#include "cuda_util.hpp"

__device__ __host__ alus::Coordinates alus::snapengine::geocoding::CrsGeocoding::GetPixelCoordinates(double x,
                                                                                                     double y) const {
    auto geo_transform = this->geo_transform_parameters_;
    double longitude = geo_transform.originLon + geo_transform.pixelSizeLon * (x) + geo_transform.pixelSizeLon / 2;
    double latitude = geo_transform.originLat + geo_transform.pixelSizeLat * (y) + geo_transform.pixelSizeLat / 2;

    return {longitude, latitude};
}
__device__ __host__ alus::Coordinates alus::snapengine::geocoding::CrsGeocoding::GetPixelCoordinates(
    std::tuple<double, double> pixel_position) const {
    return GetPixelCoordinates(std::get<0>(pixel_position), std::get<1>(pixel_position));
}
__device__ __host__ alus::Coordinates alus::snapengine::geocoding::CrsGeocoding::GetPixelCoordinates(
    alus::PixelPosition pixel_position) const {
    return GetPixelCoordinates(pixel_position.x, pixel_position.y);
}

__device__ __host__ alus::Coordinates alus::snapengine::geocoding::CrsGeocoding::GetPixelCoordinates(
    alus::PrecisePixelPosition pixel_position) const {
    return GetPixelCoordinates(pixel_position.x, pixel_position.y);
}

alus::snapengine::geocoding::CrsGeocoding::CrsGeocoding(const alus::GeoTransformParameters& geo_transform_parameters)
    : geo_transform_parameters_(geo_transform_parameters) {}

__device__ __host__ alus::PrecisePixelPosition alus::snapengine::geocoding::CrsGeocoding::GetPixelPosition(
    alus::Coordinates pixel_coordinates) const {
    return GetPixelPosition(pixel_coordinates.lon, pixel_coordinates.lat);
}
__device__ __host__ alus::PrecisePixelPosition alus::snapengine::geocoding::CrsGeocoding::GetPixelPosition(
    std::tuple<double, double> pixel_coordinates) const {
    return GetPixelPosition(std::get<0>(pixel_coordinates), std::get<1>(pixel_coordinates));
}
__device__ __host__ alus::PrecisePixelPosition alus::snapengine::geocoding::CrsGeocoding::GetPixelPosition(
    double lon, double lat) const {
    auto geo_transform = this->geo_transform_parameters_;
    double x = (lon - geo_transform.pixelSizeLon / 2 - geo_transform.originLon) / geo_transform.pixelSizeLon;
    double y = (lat - geo_transform.pixelSizeLat / 2 - geo_transform.originLat) / geo_transform.pixelSizeLat;

    return {x, y};
}
