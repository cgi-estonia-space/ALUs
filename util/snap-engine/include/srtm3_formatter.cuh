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
#include <cuda_runtime.h>
#include <device_launch_parameters.h>


namespace alus {
namespace snapengine{

struct DemFormatterData{
    double m00, m01, m02, m10, m11, m12;
    double no_data_value;
    int x_size, y_size;
    int max_lats;
    int max_lons;
    double* egm;
};

cudaError_t LaunchDemFormatter(dim3 grid_size, dim3 block_size, double *target, double *source, DemFormatterData data);

}//namespace
}//namespace
