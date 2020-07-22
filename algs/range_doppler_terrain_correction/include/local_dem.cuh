#pragma once

struct LocalDemKernelArgs {
    int dem_cols;
    int dem_rows;
    int target_cols;
    int target_rows;
    double dem_pixel_size_lon;
    double dem_pixel_size_lat;
    double dem_origin_lon;
    double dem_origin_lat;
    double target_pixel_size_lon;
    double target_pixel_size_lat;
    double target_origin_lon;
    double target_origin_lat;
};

void RunElevationKernel(double const* dem, double* target_elevations,
                        LocalDemKernelArgs const args);