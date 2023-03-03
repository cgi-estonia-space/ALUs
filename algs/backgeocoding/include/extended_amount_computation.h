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
#include "orbit_state_vector_computation.h"
#include "pointer_holders.h"
#include "s1tbx-commons/sentinel1_utils_computation.h"
#include "s1tbx-commons/subswath_info.h"
#include "s1tbx-commons/subswath_info_computation.h"
#include "shapes.h"

namespace alus::backgeocoding {

struct AzimuthAndRangeBounds {
    int azimuth_min;
    int azimuth_max;
    int range_min;
    int range_max;
};

cudaError_t LaunchComputeExtendedAmount(Rectangle bounds, AzimuthAndRangeBounds& extended_amount,
                                        snapengine::OrbitStateVectorComputation* d_orbit_state_vectors,
                                        size_t nr_of_vectors, double vectors_dt,
                                        const s1tbx::SubSwathInfo& subswath_info,
                                        s1tbx::DeviceSentinel1Utils* d_sentinel_1_utils,
                                        s1tbx::DeviceSubswathInfo* d_subswath_info, const PointerArray& tiles,
                                        const float* egm, const dem::Property* dem_property, dem::Type dem_type);
}  // namespace alus::backgeocoding
