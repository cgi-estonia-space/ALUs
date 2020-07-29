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

#include "cuda_util.cuh"
#include "orbit_state_vectors.cuh"
#include "pos_vector.h"

namespace alus {
namespace s1tbx {
namespace sargeocoding {

constexpr double NON_VALID_ZERO_DOPPLER_TIME{-99999.0};
constexpr double NON_VALID_INCIDENCE_ANGLE{-99999.0};

/**
 * Compute Doppler frequency for given earth_point and sensor position.
 *
 * @param earthPoint     The earth point in xyz coordinate.
 * @param sensorPosition Array of sensor positions for all range lines.
 * @param sensorVelocity Array of sensor velocities for all range lines.
 * @param wavelength     The radar wavelength.
 * @return The Doppler frequency in Hz.
 */
inline __device__ __host__ double getDopplerFrequency(alus::snapengine::PosVector earthPoint,
                                                      alus::snapengine::PosVector sensorPosition,
                                                      alus::snapengine::PosVector sensorVelocity,
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
 * @param first_line_utc     The zero Doppler time for the first range line.
 * @param line_time_interval The line time interval.
 * @param wavelength       The radar wavelength.
 * @param earth_point       The earth point in xyz coordinate.
 * @param sensor_position   Array of sensor positions for all range lines.
 * @param sensor_velocity   Array of sensor velocities for all range lines.
 * @return The zero Doppler time in days if it is found, -99999.0 otherwise.
 */
inline __device__ __host__ double GetEarthPointZeroDopplerTimeImpl(
    double first_line_utc,
    double line_time_interval,
    double wavelength,
    alus::snapengine::PosVector earth_point,
    cudautil::KernelArray<alus::snapengine::PosVector> sensor_position,
    cudautil::KernelArray<alus::snapengine::PosVector> sensor_velocity) {
    // binary search is used in finding the zero doppler time
    int lower_bound = 0;
    int upper_bound = static_cast<int>(sensor_position.size) - 1;
    auto lower_bound_freq = getDopplerFrequency(
        earth_point, sensor_position.array[lower_bound], sensor_velocity.array[lower_bound], wavelength);
    auto upper_bound_freq = getDopplerFrequency(
        earth_point, sensor_position.array[upper_bound], sensor_velocity.array[upper_bound], wavelength);

    if (std::abs(lower_bound_freq) < 1.0) {
        return first_line_utc + lower_bound * line_time_interval;
    } else if (std::abs(upper_bound_freq) < 1.0) {
        return first_line_utc + upper_bound * line_time_interval;
    } else if (lower_bound_freq * upper_bound_freq > 0.0) {
        return NON_VALID_ZERO_DOPPLER_TIME;
    }

    // start binary search
    double mid_freq;
    while (upper_bound - lower_bound > 1) {
        const auto mid = (int)((static_cast<double>(lower_bound) + upper_bound) / 2.0);
        mid_freq = sensor_velocity.array[mid].x * (earth_point.x - sensor_position.array[mid].x) +
                   sensor_velocity.array[mid].y * (earth_point.y - sensor_position.array[mid].y) +
                   sensor_velocity.array[mid].z * (earth_point.z - sensor_position.array[mid].z);

        if (mid_freq * lower_bound_freq > 0.0) {
            lower_bound = mid;
            lower_bound_freq = mid_freq;
        } else if (mid_freq * upper_bound_freq > 0.0) {
            upper_bound = mid;
            upper_bound_freq = mid_freq;
        } else if (mid_freq == 0.0) {
            return first_line_utc + mid * line_time_interval;
        }
    }

    const auto y0 =
        lower_bound - lower_bound_freq * (upper_bound - lower_bound) / (upper_bound_freq - lower_bound_freq);
    return first_line_utc + y0 * line_time_interval;
}

inline __device__ __host__ double ComputeSlantRangeImpl(double time,
                                                        cudautil::KernelArray<snapengine::OrbitStateVector> vectors,
                                                        snapengine::PosVector earth_point,
                                                        snapengine::PosVector& sensor_pos) {
    sensor_pos = orbitstatevectors::GetPositionImpl(time, vectors);
    double const xDiff = sensor_pos.x - earth_point.x;
    double const yDiff = sensor_pos.y - earth_point.y;
    double const zDiff = sensor_pos.z - earth_point.z;

    return std::sqrt(xDiff * xDiff + yDiff * yDiff + zDiff * zDiff);
}

inline __device__ __host__ bool IsDopplerTimeValidImpl(double first_line_utc,
                                                       double last_line_utc,
                                                       double zero_doppler_time) {
    return zero_doppler_time >= std::min(first_line_utc, last_line_utc) &&
           zero_doppler_time <= std::max(first_line_utc, last_line_utc);
}

inline __device__ __host__ double ComputeRangeIndexSlcImpl(double range_spacing,
                                                           double slant_range,
                                                           double near_edge_slant_range) {
    return (slant_range - near_edge_slant_range) / range_spacing;
}

inline __device__ __host__ bool IsValidCellImpl(
    double range_index, double azimuth_index, int diff_lat, int src_max_range, int src_max_azimuth) {
    if (range_index < 0.0 || range_index >= src_max_range || azimuth_index <= 0.0 | azimuth_index >= src_max_azimuth) {
        return false;
    }

    return diff_lat < 5;
}

}  // namespace sargeocoding
}  // namespace s1tbx
}  // namespace alus