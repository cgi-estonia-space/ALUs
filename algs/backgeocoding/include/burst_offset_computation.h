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
#include "orbit_state_vector_computation.h"
#include "pointer_holders.h"
#include "s1tbx-commons/sentinel1_utils_computation.h"
#include "s1tbx-commons/subswath_info_computation.h"

namespace alus::backgeocoding {

struct BurstOffsetKernelArgs {
    double* latitudes;
    double* longitudes;
    PointerArray srtm3_tiles;
    const dem::Property* dem_property_;
    int* burst_offset;

    const s1tbx::DeviceSubswathInfo* master_subswath_info;
    const s1tbx::DeviceSentinel1Utils* master_sentinel_utils;
    snapengine::OrbitStateVectorComputation* master_orbit;
    int master_num_orbit_vec;
    double master_dt;

    const s1tbx::DeviceSubswathInfo* slave_subswath_info;
    const s1tbx::DeviceSentinel1Utils* slave_sentinel_utils;
    snapengine::OrbitStateVectorComputation* slave_orbit;
    int slave_num_orbit_vec;
    double slave_dt;

    size_t width;
    size_t height;
};

cudaError_t LaunchBurstOffsetKernel(BurstOffsetKernelArgs& args, int* burst_offset);
}  // namespace alus::backgeocoding
