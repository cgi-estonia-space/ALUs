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

#include "general_constants.h"
#include "orbit_state_vector_computation.h"
#include "pointer_holders.h"
#include "position_data.h"
#include "s1tbx-commons/sentinel1_utils_computation.h"
#include "s1tbx-commons/subswath_info_computation.h"

namespace alus::backgeocoding {

struct SlavePixPosData {
    int num_lines;
    int num_pixels;

    int m_burst_index;
    int s_burst_index;

    int lat_max_idx;
    int lat_min_idx;
    int lon_min_idx;
    int lon_max_idx;

    PointerArray tiles;

    // earth gravitational model
    const float* egm;

    int max_lats;
    int max_lons;
    double dem_no_data_value;
    bool mask_out_area_without_elevation;

    size_t* device_valid_index_counter;
    double *device_master_az, *device_master_rg;
    double *device_slave_az, *device_slave_rg;
    double *device_lats, *device_lons;

    s1tbx::DeviceSubswathInfo* device_master_subswath;
    s1tbx::DeviceSubswathInfo* device_slave_subswath;

    s1tbx::DeviceSentinel1Utils* device_master_utils;
    s1tbx::DeviceSentinel1Utils* device_slave_utils;

    snapengine::OrbitStateVectorComputation* device_master_orbit_state_vectors;
    snapengine::OrbitStateVectorComputation* device_slave_orbit_state_vectors;
    int nr_of_master_vectors, nr_of_slave_vectors;

    double master_dt, slave_dt;
};

cudaError_t LaunchSlavePixPos(SlavePixPosData calc_data);
cudaError_t LaunchFillXAndY(double* device_x_points, double* device_y_points, size_t points_size,
                            double placeholder_value);

}  // namespace alus::backgeocoding
