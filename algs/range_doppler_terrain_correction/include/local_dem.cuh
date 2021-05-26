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