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

#include <cstdint>

namespace alus {
namespace snapengine{

// TODO: Actual geotransform values are double types.
struct Srtm3FormatComputation {
    float m00; // x-dimension of a pixel in map units
    float m01; // rotation
    float m02; // x-coordinate of center of upper left pixel
    float m10; // rotation
    float m11; // NEGATIVE of y-dimension of a pixel in map units
    float m12; // y-coordinate of center of upper left pixel
    int16_t no_data_value;
    int x_size, y_size;
    int max_lats;
    int max_lons;
    float* egm;
};

cudaError_t LaunchDemFormatter(dim3 grid_size, dim3 block_size, float *target, float *source,
                               Srtm3FormatComputation data);

}//namespace
}//namespace
