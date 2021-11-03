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

#include "s1tbx-commons/sar_geocoding.h"

#include "s1tbx-commons/sar_geocoding.cuh"

namespace alus {
namespace s1tbx {
namespace sargeocoding {

double GetEarthPointZeroDopplerTime(double first_line_utc,
                                    double line_time_interval,
                                    double wavelength,
                                    alus::snapengine::PosVector earth_point,
                                    cuda::KernelArray<alus::snapengine::PosVector> sensor_position,
                                    cuda::KernelArray<alus::snapengine::PosVector> sensor_velocity) {
    return GetEarthPointZeroDopplerTimeImpl(
        first_line_utc, line_time_interval, wavelength, earth_point, sensor_position, sensor_velocity);
}

double ComputeSlantRange(double time,
                         cuda::KernelArray<snapengine::OrbitStateVectorComputation> vectors,
                         snapengine::PosVector earth_point,
                         snapengine::PosVector& sensor_pos) {
    return ComputeSlantRangeImpl(time, vectors, earth_point, sensor_pos);
}

bool IsDopplerTimeValid(double first_line_utc, double last_line_utc, double zero_doppler_time) {
    return IsDopplerTimeValidImpl(first_line_utc, last_line_utc, zero_doppler_time);
}

double ComputeRangeIndexSlc(double range_spacing, double slant_range, double near_edge_slant_range) {
    return ComputeRangeIndexSlcImpl(range_spacing, slant_range, near_edge_slant_range);
}

bool IsValidCell(double range_index, double azimuth_index, int diff_lat, int src_max_range, int src_max_azimuth) {
    return IsValidCellImpl(range_index, azimuth_index, diff_lat, src_max_range, src_max_azimuth);
}

}  // namespace sargeocoding
}  // namespace s1tbx
}  // namespace alus