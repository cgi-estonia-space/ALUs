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

#include "geo_utils.cuh"
#include "get_position.h"
#include "position_data.h"
#include "sar_geocoding.cuh"

// TODO: remove these MACROS
#define DEBUG printf("Reached here => %s:%d\n", __FILE__, __LINE__)

namespace alus {
namespace terraincorrection {

inline __device__ __host__ bool GetPositionImpl(
    double lat, double lon, double alt, s1tbx::PositionData& satellite_pos, const GetPositionMetadata& metadata) {
    snapengine::geoutils::Geo2xyzWgs84Impl(lat, lon, alt, satellite_pos.earth_point);
    const auto zero_doppler_time = s1tbx::sargeocoding::GetEarthPointZeroDopplerTimeImpl(metadata.first_line_utc,
                                                                                         metadata.line_time_interval,
                                                                                         metadata.wavelength,
                                                                                         satellite_pos.earth_point,
                                                                                         metadata.sensor_position,
                                                                                         metadata.sensor_velocity);
    if (zero_doppler_time == s1tbx::sargeocoding::NON_VALID_ZERO_DOPPLER_TIME) {
        return false;
    }
    satellite_pos.slant_range = s1tbx::sargeocoding::ComputeSlantRangeImpl(
        zero_doppler_time, metadata.orbit_state_vector, satellite_pos.earth_point, satellite_pos.sensor_pos);

    satellite_pos.range_index = s1tbx::sargeocoding::ComputeRangeIndexSlcImpl(
        metadata.range_spacing,
        satellite_pos.slant_range,
        metadata.near_edge_slant_range);

    if (satellite_pos.range_index == -1.0) {
        return false;
    }

    satellite_pos.azimuth_index = (zero_doppler_time - metadata.first_line_utc) / metadata.line_time_interval;

    return true;
}
}  // namespace terraincorrection
}  // namespace alus