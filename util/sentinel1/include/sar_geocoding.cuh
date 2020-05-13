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

#include <cmath>

#include "PosVector.hpp"
#include "cuda_util.cuh"

namespace slap {
namespace s1tbx {
namespace sarGeocoding {

/**
 * Compute Doppler frequency for given earthPoint and sensor position.
 *
 * @param earthPoint     The earth point in xyz coordinate.
 * @param sensorPosition Array of sensor positions for all range lines.
 * @param sensorVelocity Array of sensor velocities for all range lines.
 * @param wavelength     The radar wavelength.
 * @return The Doppler frequency in Hz.
 */
inline __device__ __host__ double getDopplerFrequency(snapEngine::PosVector earthPoint,
                                                      snapEngine::PosVector sensorPosition,
                                                      snapEngine::PosVector sensorVelocity,
                                                      double wavelength) {
    const auto xDiff = earthPoint.x - sensorPosition.x;
    const auto yDiff = earthPoint.y - sensorPosition.y;
    const auto zDiff = earthPoint.z - sensorPosition.z;
    const auto distance = sqrt(xDiff * xDiff + yDiff * yDiff + zDiff * zDiff);

    return 2.0 * (sensorVelocity.x * xDiff + sensorVelocity.y * yDiff + sensorVelocity.z * zDiff) /
           (distance * wavelength);
}

/**
 * Compute zero Doppler time for given earth point using bisection method.
 *
 * Duplicate of a SNAP's SARGeocoding.java's getEarthPointZeroDopplerTime().
 *
 * @param firstLineUTC     The zero Doppler time for the first range line.
 * @param lineTimeInterval The line time interval.
 * @param wavelength       The radar wavelength.
 * @param earthPoint       The earth point in xyz coordinate.
 * @param sensorPosition   Array of sensor positions for all range lines.
 * @param sensorVelocity   Array of sensor velocities for all range lines.
 * @return The zero Doppler time in days if it is found, -99999.0 otherwise.
 */
inline __device__ __host__ double GetEarthPointZeroDopplerTime_impl(double firstLineUTC,
                                                                    double lineTimeInterval,
                                                                    double wavelength,
                                                                    snapEngine::PosVector earthPoint,
                                                                    KernelArray<snapEngine::PosVector> sensorPosition,
                                                                    KernelArray<snapEngine::PosVector> sensorVelocity) {
    // binary search is used in finding the zero doppler time
    int lowerBound = 0;
    int upperBound = static_cast<int>(sensorPosition.size) - 1;
    auto lowerBoundFreq =
        getDopplerFrequency(earthPoint, sensorPosition.array[lowerBound], sensorVelocity.array[lowerBound], wavelength);
    auto upperBoundFreq =
        getDopplerFrequency(earthPoint, sensorPosition.array[upperBound], sensorVelocity.array[upperBound], wavelength);

    if (std::abs(lowerBoundFreq) < 1.0) {
        return firstLineUTC + lowerBound * lineTimeInterval;
    } else if (std::abs(upperBoundFreq) < 1.0) {
        return firstLineUTC + upperBound * lineTimeInterval;
    } else if (lowerBoundFreq * upperBoundFreq > 0.0) {
        return -99999.0;
    }

    // start binary search
    double midFreq;
    while (upperBound - lowerBound > 1) {
        const auto mid = (int)((static_cast<double>(lowerBound) + upperBound) / 2.0);
        midFreq = sensorVelocity.array[mid].x * (earthPoint.x - sensorPosition.array[mid].x) +
                  sensorVelocity.array[mid].y * (earthPoint.y - sensorPosition.array[mid].y) +
                  sensorVelocity.array[mid].z * (earthPoint.z - sensorPosition.array[mid].z);

        if (midFreq * lowerBoundFreq > 0.0) {
            lowerBound = mid;
            lowerBoundFreq = midFreq;
        } else if (midFreq * upperBoundFreq > 0.0) {
            upperBound = mid;
            upperBoundFreq = midFreq;
        } else if (midFreq == 0.0) {
            return firstLineUTC + mid * lineTimeInterval;
        }
    }

    const auto y0 = lowerBound - lowerBoundFreq * (upperBound - lowerBound) / (upperBoundFreq - lowerBoundFreq);
    return firstLineUTC + y0 * lineTimeInterval;
}
}  // namespace sarGeocoding
}  // namespace s1tbx
}  // namespace slap