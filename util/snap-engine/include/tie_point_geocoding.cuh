#pragma once

#include <cuda_runtime.h>

#include "geocoding.cuh"
#include "raster_properties.hpp"
#include "tie_point_grid.cuh"
#include "tie_point_grid.h"

namespace alus {
namespace snapengine {
namespace geocoding {

/**
 * Copy of SNAP class TiePointGeocoding
 *
 * @todo Complete documentation
 */
class TiePointGeocoding : public alus::snapengine::geocoding::Geocoding {
   public:
    __device__ __host__ Coordinates GetPixelCoordinates(PixelPosition pixel_position) const override;
    __device__ __host__ Coordinates GetPixelCoordinates(PrecisePixelPosition pixel_position) const override;
    __device__ __host__ Coordinates GetPixelCoordinates(std::tuple<double, double> pixel_position) const override;
    __device__ __host__ Coordinates GetPixelCoordinates(double x, double y) const override;

    __device__ __host__ PrecisePixelPosition GetPixelPosition(Coordinates pixel_coordinates) const override;
    __device__ __host__ PrecisePixelPosition
    GetPixelPosition(std::tuple<double, double> pixel_coordinates) const override;
    __device__ __host__ PrecisePixelPosition GetPixelPosition(double lon, double lat) const override;

    TiePointGeocoding(tiepointgrid::TiePointGrid latitude_grid, tiepointgrid::TiePointGrid longitude_grid)
        : latitude_grid_(latitude_grid), longitude_grid_(longitude_grid){};

    alus::snapengine::tiepointgrid::TiePointGrid latitude_grid_;
    alus::snapengine::tiepointgrid::TiePointGrid longitude_grid_;

   private:
    bool approximations_computed_ = false;
};
}  // namespace geocoding
}  // namespace snapengine
}  // namespace alus