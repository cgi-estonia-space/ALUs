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

#include "orbit_state_vector_computation.h"
#include "pointer_holders.h"
#include "sentinel1_utils.cuh"
#include "shapes.h"
#include "subswath_info.h"

namespace alus {
namespace backgeocoding {
struct AzimuthAndRangeBounds {
    double azimuth_min;
    double azimuth_max;
    double range_min;
    double range_max;
};

cudaError_t LaunchComputeExtendedAmount(Rectangle bounds,
                                        AzimuthAndRangeBounds &extended_amount,
                                        const snapengine::OrbitStateVectorComputation* vectors,
                                        size_t nr_of_vectors,
                                        double vectors_dt,
                                        const s1tbx::SubSwathInfo& subswath_info,
                                        s1tbx::DeviceSentinel1Utils* d_sentinel_1_utils,
                                        s1tbx::DeviceSubswathInfo* d_subswath_info,
                                        const PointerArray &tiles,
                                        float *egm);
}  // namespace backgeocoding
}  // namespace alus