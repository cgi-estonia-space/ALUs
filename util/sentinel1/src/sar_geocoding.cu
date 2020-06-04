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

#include "sar_geocoding.h"

#include "sar_geocoding.cuh"

namespace alus {
namespace s1tbx {
namespace sarGeocoding {
double GetEarthPointZeroDopplerTime(double firstLineUTC,
                                    double lineTimeInterval,
                                    double wavelength,
                                    alus::snapengine::PosVector earthPoint,
                                    KernelArray<alus::snapengine::PosVector> sensorPosition,
                                    KernelArray<alus::snapengine::PosVector> sensorVelocity) {
    return GetEarthPointZeroDopplerTime_impl(
        firstLineUTC, lineTimeInterval, wavelength, earthPoint, sensorPosition, sensorVelocity);
}

double ComputeSlantRange(double time,
                         KernelArray<snapengine::OrbitStateVector> vectors,
                         snapengine::PosVector earthPoint,
                         snapengine::PosVector& sensorPos) {
    return ComputeSlantRangeImpl(time, vectors, earthPoint, sensorPos);
}

bool IsDopplerTimeValid(double first_line_utc, double last_line_utc, double zero_doppler_time) {
    return IsDopplerTimeValidImpl(first_line_utc, last_line_utc, zero_doppler_time);
}

double ComputeRangeIndexSlc(double range_spacing, double slant_range, double near_edge_slant_range) {
    return ComputeRangeIndexSlcImpl(range_spacing, slant_range, near_edge_slant_range);
}

}  // namespace sarGeocoding
}  // namespace s1tbx
}  // namespace alus