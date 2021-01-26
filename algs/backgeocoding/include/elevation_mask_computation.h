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

#include "pointer_holders.h"

namespace alus {
namespace backgeocoding {

struct ElevationMaskData {

    size_t size; //all 4 of the following arrays have the same size, which is in here.
    double *device_x_points;
    double *device_y_points;
    double *device_lat_array;
    double *device_lon_array;

    PointerArray tiles;

};

cudaError_t LaunchElevationMask(ElevationMaskData data);

}  // namespace backgeocoding
}  // namespace alus
