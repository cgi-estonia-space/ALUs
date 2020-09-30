#pragma once

#include "raster_properties.hpp"

struct LocalDemKernelArgs {
    double dem_x_0;
    double dem_y_0;
    int dem_tile_width;
    int dem_tile_height;
    double target_x_0;
    double target_y_0;
    int target_width;
    int target_height;
    double dem_no_data_value;
    alus::GeoTransformParameters dem_geo_transform;
    alus::GeoTransformParameters target_geo_transform;
};

void RunElevationKernel(double const* dem, double* target_elevations,
                        LocalDemKernelArgs const args);