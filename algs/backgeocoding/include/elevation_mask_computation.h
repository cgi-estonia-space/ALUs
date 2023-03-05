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

#include <driver_types.h>

#include "dem_property.h"
#include "dem_type.h"
#include "pointer_holders.h"

namespace alus::backgeocoding {

struct ElevationMaskData {
    size_t size;  // all 4 of the following arrays have the same size, which is in here.
    int* not_null_counter;
    double* device_x_points;
    double* device_y_points;
    double* device_lat_array;
    double* device_lon_array;
    bool mask_out_area_without_elevation;
    PointerArray tiles;
    const dem::Property* dem_property;
    dem::Type dem_type;
};

cudaError_t LaunchElevationMask(ElevationMaskData data, cudaStream_t stream);

}  // namespace alus::backgeocoding
