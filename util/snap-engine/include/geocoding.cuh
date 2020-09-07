#pragma once

#include <cuda_runtime.h>
#include <tuple>

#include "raster_properties.hpp"

namespace alus {
namespace snapengine {
namespace geocoding {
class Geocoding {
   public:
    __device__ __host__ virtual Coordinates GetPixelCoordinates(PixelPosition pixel_position) const = 0;
    __device__ __host__ virtual Coordinates GetPixelCoordinates(PrecisePixelPosition pixel_position) const = 0;
    __device__ __host__ virtual Coordinates GetPixelCoordinates(std::tuple<double, double> pixel_position) const = 0;
    __device__ __host__ virtual Coordinates GetPixelCoordinates(double x, double y) const = 0;

    __device__ __host__ virtual PrecisePixelPosition GetPixelPosition(Coordinates pixel_coordinates) const = 0;
    __device__ __host__ virtual PrecisePixelPosition GetPixelPosition(std::tuple<double, double> pixel_coordinates) const = 0;
    __device__ __host__ virtual PrecisePixelPosition GetPixelPosition(double lon, double lat) const = 0;
};
}  // namespace geocoding
}  // namespace snapengine
}  // namespace alus