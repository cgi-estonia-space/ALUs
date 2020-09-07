#pragma once

#include "geocoding.cuh"

namespace alus {
namespace snapengine {
namespace geocoding {
class CrsGeocoding : public alus::snapengine::geocoding::Geocoding {
   public:
    __device__ __host__ Coordinates GetPixelCoordinates(PixelPosition pixel_position) const override;
    __device__ __host__ Coordinates GetPixelCoordinates(PrecisePixelPosition pixel_position) const override;
    __device__ __host__ Coordinates GetPixelCoordinates(std::tuple<double, double> pixel_position) const override;
    __device__ __host__ Coordinates GetPixelCoordinates(double x, double y) const override;

    __device__ __host__ PrecisePixelPosition GetPixelPosition(Coordinates pixel_coordinates) const override;
    __device__ __host__ PrecisePixelPosition GetPixelPosition(std::tuple<double, double> pixel_coordinates) const override;
    __device__ __host__ PrecisePixelPosition GetPixelPosition(double lon, double lat) const override;

    CrsGeocoding(const GeoTransformParameters& geo_transform_parameters);

    GeoTransformParameters geo_transform_parameters_;
};
}  // namespace geocoding
}  // namespace snapengine
}  // namespace alus