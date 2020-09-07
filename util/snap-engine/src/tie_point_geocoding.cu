#include "tie_point_geocoding.cuh"

__device__ __host__ alus::Coordinates alus::snapengine::geocoding::TiePointGeocoding::GetPixelCoordinates(
    alus::PixelPosition pixel_position) const {
    return this->GetPixelCoordinates(pixel_position.x + 0.5, pixel_position.y + 0.5);
}
__device__ __host__ alus::Coordinates alus::snapengine::geocoding::TiePointGeocoding::GetPixelCoordinates(
    std::tuple<double, double> pixel_position) const {
    return this->GetPixelCoordinates(std::get<0>(pixel_position), std::get<1>(pixel_position));
}
__device__ __host__ alus::Coordinates alus::snapengine::geocoding::TiePointGeocoding::GetPixelCoordinates(
    double x, double y) const {
    double latitude = alus::snapengine::tiepointgrid::GetPixelDoubleImpl(x, y, &this->latitude_grid_);
    double longitude = alus::snapengine::tiepointgrid::GetPixelDoubleImpl(x, y, &this->longitude_grid_);
    return {longitude, latitude};
}

__device__ __host__ alus::Coordinates alus::snapengine::geocoding::TiePointGeocoding::GetPixelCoordinates(
    alus::PrecisePixelPosition pixel_position) const {
  return this->GetPixelCoordinates(pixel_position.x, pixel_position.y);
}

#define UNUSED(x) (void)(x)

__device__ __host__ alus::PrecisePixelPosition alus::snapengine::geocoding::TiePointGeocoding::GetPixelPosition(
    alus::Coordinates pixel_coordinates) const {
    UNUSED(pixel_coordinates);



    return {};
}
__device__ __host__ alus::PrecisePixelPosition alus::snapengine::geocoding::TiePointGeocoding::GetPixelPosition(
    std::tuple<double, double> pixel_coordinates) const {
    UNUSED(pixel_coordinates);
    // TODO: IMPOSSIBLE TO IMPLEMENT
    return {};
}
__device__ __host__ alus::PrecisePixelPosition alus::snapengine::geocoding::TiePointGeocoding::GetPixelPosition(
    double lon, double lat) const {
    // TODO: IMPOSSIBLE TO IMPLEMENT
    UNUSED(lon);
    UNUSED(lat);
    return {};
}



