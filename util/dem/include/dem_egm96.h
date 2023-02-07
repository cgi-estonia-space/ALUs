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

namespace alus::dem {

struct EgmFormatProperties {
    double m00;  // x-dimension of a pixel in map units
    double m01;  // rotation
    double m02;  // x-coordinate of center of upper left pixel
    double m10;  // rotation
    double m11;  // NEGATIVE of y-dimension of a pixel in map units
    double m12;  // y-coordinate of center of upper left pixel
    float no_data_value;
    int tile_size_x;
    int tile_size_y;
    int grid_max_lat;
    int grid_max_lon;
    const float* device_egm_array;
};

void ConditionWithEgm96(dim3 grid_size, dim3 block_size, float* device_buffer_conditioned, float* device_buffer_dem,
                        EgmFormatProperties prop);

}  // namespace alus::dem