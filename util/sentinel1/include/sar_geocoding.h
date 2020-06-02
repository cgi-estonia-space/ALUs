/**
 * This file is a filtered duplicate of a SNAP's SARGeocoding.java ported for native code.
 * Copied from a snap-engine's(https://github.com/senbox-org/s1tbx) repository originally stated to be implemented
 * by "Copyright (C) 2016 by Array Systems Computing Inc. http://www.array.ca"
 *
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

#include <thrust/device_vector.h>

#include "PosVector.hpp"
#include "cuda_util.cuh"
#include "orbit_state_vectors.h"

namespace alus {
namespace s1tbx {
namespace sarGeocoding {
/**
 * Compute zero Doppler time for given earth point using bisection method.
 *
 * Duplicate of a SNAP's SARGeocoding.java's getEarthPointZeroDopplerTime().
 * This actually exists as an inline version for CUDA calls as GetEarthPointZeroDopplerTime_impl() in sar_geocoding.cuh
 * This procedure is duplicated by the nvcc for host processing in sar_geocoding.cu.
 *
 * @param firstLineUTC     The zero Doppler time for the first range line.
 * @param lineTimeInterval The line time interval.
 * @param wavelength       The radar wavelength.
 * @param earthPoint       The earth point in xyz coordinate.
 * @param sensorPosition   Array of sensor positions for all range lines.
 * @param sensorVelocity   Array of sensor velocities for all range lines.
 * @return The zero Doppler time in days if it is found, -1 otherwise.
 */
double GetEarthPointZeroDopplerTime(double firstLineUTC,
                                    double lineTimeInterval,
                                    double wavelength,
                                    alus::snapengine::PosVector earthPoint,
                                    KernelArray<alus::snapengine::PosVector> sensorPosition,
                                    KernelArray<alus::snapengine::PosVector> sensorVelocity);

/**
 * Compute slant range distance for given earth point and given time.
 *
 * Duplicate of a SNAP's SARGeocoding.java's computeSlantRange().
 * This actually exists as an inline version for CUDA calls as ComputeSlantRangeImpl() in sar_geocoding.cuh.
 * This procedure is duplicated by the nvcc for host processing in sar_geocoding.cu.
 *
 * @param time       The given time in days.
 * @param vectors    Orbit state vectors for getPosition calculation happening inside this function.
 * @param earthPoint The earth point in xyz coordinate.
 * @param sensorPos  The sensor position which is getting value of orbitstatevectors::GetPosition()
 * @return The slant range distance in meters.
 */
double ComputeSlantRange(double time,
                         KernelArray<snapengine::OrbitStateVector> vectors,
                         snapengine::PosVector earthPoint,
                         snapengine::PosVector& sensorPos);

/** Validates that Zero Doppler Time is between UTC timestamps.
 *
 * This is extracted from SARGeocoding.java class where it is used as part of computeRangeIndex() and
 * computeExtendedRangeIndex().
 * This actually exists as an inline version for CUDA calls as IsDopplerTimeValidImpl() in sar_geocoding.cuh.
 * This procedure is duplicated by the nvcc for host processing in sar_geocoding.cu.
 *
 * @return true when zero_doppler_time value is between first and last line UTC times (inclusive)
 */
bool IsDopplerTimeValid(double first_line_utc, double last_line_utc, double zero_doppler_time);

/**
 * Compute range index in source image for earth point with given zero Doppler time and slant range.
 *
 * This is adopted from SNAP's SARGeocoding::computeRangeIndex() with only SLC part implemented and stripped
 * doppler time validation separated as IsDopplerTimeValid().
 * This actually exists as an inline version for CUDA calls as ComputeRangeIndexSlcImpl() in sar_geocoding.cuh.
 * This procedure is duplicated by the nvcc for host processing in sar_geocoding.cu.
 *
 * @param zeroDopplerTime The zero Doppler time in MJD.
 * @param slantRange      The slant range in meters.
 * @return The range index.
 */
double ComputeRangeIndexSlc(double range_spacing, double slant_range, double near_edge_slant_range);

}  // namespace sarGeocoding
}  // namespace s1tbx
}  // namespace alus