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

namespace slap {
namespace s1tbx {
namespace sarGeocoding {
double GetEarthPointZeroDopplerTime(double firstLineUTC,
                                    double lineTimeInterval,
                                    double wavelength,
                                    snapEngine::PosVector earthPoint,
                                    KernelArray<snapEngine::PosVector> sensorPosition,
                                    KernelArray<snapEngine::PosVector> sensorVelocity) {
    return GetEarthPointZeroDopplerTime_impl(
        firstLineUTC, lineTimeInterval, wavelength, earthPoint, sensorPosition, sensorVelocity);
}
}  // namespace sarGeocoding
}  // namespace s1tbx
}  // namespace slap