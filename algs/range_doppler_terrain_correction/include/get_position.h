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

#include "kernel_array.h"
#include "orbit_state_vector_computation.h"
#include "pos_vector.h"
#include "position_data.h"

namespace alus {
namespace terraincorrection {

struct GetPositionMetadata {
    double first_line_utc;
    double line_time_interval;
    double wavelength;
    double range_spacing;
    double near_edge_slant_range;
    cuda::KernelArray<snapengine::PosVector> sensor_position;
    cuda::KernelArray<snapengine::PosVector> sensor_velocity;
    cuda::KernelArray<snapengine::OrbitStateVectorComputation> orbit_state_vectors;
};

/**
 * Determines satellite position by filling PositionData structure.
 *
 * This is a port from RangeDopplerGeoCodingOp.java's GetPosition() procedure. It consists of 4 calculations -
 * geoutils::Geo2xyzWgs84(), SarGeocoding::getEarthPointZeroDopplerTime(), SarGeocoding::computeSlantRange,
 * SarGeocoding::computeRangeIndex() [only SLC images].
 * All class fields and static data is supplied here as a separate argument.
 *
 * @param[in] lat Pixel latitude
 * @param[in] lon Pixel longitude
 * @param[in] alt Pixel elevation (in SNAP code it is altitude, but elevation is more precise)
 * @param[out] satellite_pos Structure that consists of satellite position
 * @param[in] metadata Various metadata needed for the calulcations.
 * @return
 */
bool GetPosition(double lat, double lon, double alt, s1tbx::PositionData& satellite_pos, const GetPositionMetadata& metadata);

}  // namespace terraincorrection
}  // namespace alus
