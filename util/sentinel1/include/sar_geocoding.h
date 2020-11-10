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

#include "cuda_util.cuh"
#include "orbit_state_vector_computation.h"
#include "pos_vector.h"

namespace alus {
namespace s1tbx {
namespace sargeocoding {
/**
 * Compute zero Doppler time for given earth point using bisection method.
 *
 * Duplicate of a SNAP's SARGeocoding.java's getEarthPointZeroDopplerTime().
 * This actually exists as an inline version for CUDA calls as GetEarthPointZeroDopplerTimeImpl() in sar_geocoding.cuh
 * This procedure is duplicated by the nvcc for host processing in sar_geocoding.cu.
 *
 * @param first_line_utc     The zero Doppler time for the first range line.
 * @param line_time_interval The line time interval.
 * @param wavelength       The radar wavelength.
 * @param earth_point       The earth point in xyz coordinate.
 * @param sensor_position   Array of sensor positions for all range lines.
 * @param sensor_velocity   Array of sensor velocities for all range lines.
 * @return The zero Doppler time in days if it is found, -1 otherwise.
 */
double GetEarthPointZeroDopplerTime(double first_line_utc,
                                    double line_time_interval,
                                    double wavelength,
                                    alus::snapengine::PosVector earth_point,
                                    cuda::KernelArray<alus::snapengine::PosVector> sensor_position,
                                    cuda::KernelArray<alus::snapengine::PosVector> sensor_velocity);

/**
 * Compute slant range distance for given earth point and given time.
 *
 * Duplicate of a SNAP's SARGeocoding.java's computeSlantRange().
 * This actually exists as an inline version for CUDA calls as ComputeSlantRangeImpl() in sar_geocoding.cuh.
 * This procedure is duplicated by the nvcc for host processing in sar_geocoding.cu.
 *
 * @param time       The given time in days.
 * @param vectors    Orbit state vectors for GetPosition calculation happening inside this function.
 * @param earth_point The earth point in xyz coordinate.
 * @param sensor_pos  The sensor position which is getting value of orbitstatevectors::GetPosition()
 * @return The slant range distance in meters.
 */
double ComputeSlantRange(double time,
                         cuda::KernelArray<snapengine::OrbitStateVectorComputation> vectors,
                         snapengine::PosVector earth_point,
                         snapengine::PosVector& sensor_pos);

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

/**
 * Determines if tile is located inside the given input area.
 *
 * @param range_index
 * @param azimuth_index
 * @param diff_lat
 * @param src_max_range
 * @param src_max_azimuth
 * @return true if the cell is valid and false otherwise
 * @todo part of the logic is not implemented as it is only used for very long images such as GM, WSM or assembled slices which is not the case in test data provided
 */
bool IsValidCell(double range_index, double azimuth_index, int diff_lat, int src_max_range, int src_max_azimuth);

}  // namespace sargeocoding
}  // namespace s1tbx
}  // namespace alus