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

#include "../../snap-engine/pos_vector.h"
#include "cuda_util.cuh"
#include "s1tbx-commons/orbit_state_vectors.cuh"

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
 * Compute Doppler frequency for given earthPoint and sensor position.
 *
 * @param earthPoint     The earth point in xyz coordinate.
 * @param orbit          OrbitStateVector
 * @param wavelength     The radar wavelength.
 * @return The Doppler frequency in Hz.
 */
inline __device__ __host__ double getDopplerFrequency(alus::snapengine::PosVector earthPoint,
                                                      alus::snapengine::OrbitStateVectorComputation orbit,
                                                      double wavelength) {
    double xDiff = earthPoint.x - orbit.xPos_;
    double yDiff = earthPoint.y - orbit.yPos_;
    double zDiff = earthPoint.z - orbit.zPos_;
    double distance = sqrt(xDiff * xDiff + yDiff * yDiff + zDiff * zDiff);

    return 2.0 * (orbit.xVel_ * xDiff + orbit.yVel_ * yDiff + orbit.zVel_ * zDiff) / (distance * wavelength);
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
    cuda::KernelArray<alus::snapengine::PosVector> sensor_position,
    cuda::KernelArray<alus::snapengine::PosVector> sensor_velocity) {
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
        const auto mid = (int)((lower_bound + upper_bound) / 2.0);
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

/**
 * Compute zero Doppler time for given point with the product orbit state vectors using bisection method.
 *
 * @param line_time_interval The line time interval.
 * @param wavelength       The radar wavelength.
 * @param earth_point       The earth point in xyz coordinate.
 * @param orbit            The array of orbit state vectors.
 * @return The zero Doppler time in days if it is found, NonValidZeroDopplerTime otherwise.
 */

inline __device__ double GetZeroDopplerTimeLUT(double line_time_interval,
                                            double wavelength,
                                            snapengine::PosVector earth_point,
                                            snapengine::OrbitStateVectorComputation* orbit,
                                            const int num_orbit_vec,
                                            const double dt, const double* osv_lut) {
    double first_vec_time = 0.0;
    double second_vec_time = 0.0;
    double first_vec_freq = 0.0;
    double second_vec_freq = 0.0;

    for (int i = 0; i < num_orbit_vec; i++) {
        snapengine::OrbitStateVectorComputation orb = orbit[i];

        double current_freq = getDopplerFrequency(earth_point, orb, wavelength);
        if (i == 0 || first_vec_freq * current_freq > 0) {
            first_vec_time = orb.timeMjd_;
            first_vec_freq = current_freq;
        } else {
            second_vec_time = orb.timeMjd_;
            second_vec_freq = current_freq;
            break;
        }
    }

    if (first_vec_freq * second_vec_freq >= 0.0) {
        return NON_VALID_ZERO_DOPPLER_TIME;
    }

    // find the exact time using Doppler frequency and bisection method
    double lower_bound_time = first_vec_time;
    double upper_bound_time = second_vec_time;
    double lower_bound_freq = first_vec_freq;
    double upper_bound_freq = second_vec_freq;
    double mid_time, mid_freq;
    double diff_time = std::abs(upper_bound_time - lower_bound_time);
    const double abs_line_time_interval = std::abs(line_time_interval);

    const int total_iterations = (int)(diff_time / abs_line_time_interval) + 1;
    int num_iterations = 0;
    while (diff_time > abs_line_time_interval && num_iterations <= total_iterations) {
        mid_time = (upper_bound_time + lower_bound_time) / 2.0;

        orbitstatevectors::PositionVelocity posvel =
            orbitstatevectors::GetPositionVelocityLUT(mid_time, orbit, num_orbit_vec, dt, osv_lut);
        mid_freq = getDopplerFrequency(earth_point, posvel.position, posvel.velocity, wavelength);

        if (mid_freq * lower_bound_freq > 0.0) {
            lower_bound_time = mid_time;
            lower_bound_freq = mid_freq;
        } else if (mid_freq * upper_bound_freq > 0.0) {
            upper_bound_time = mid_time;
            upper_bound_freq = mid_freq;
            // TODO: there might be an accuracy bug here because we are missing a compare delta, but what is the delta?
        } else if (mid_freq == 0.0) {
            return mid_time;
        }

        diff_time = std::abs(upper_bound_time - lower_bound_time);
        num_iterations++;
    }

    return lower_bound_time - lower_bound_freq * (upper_bound_time - lower_bound_time) / (upper_bound_freq - lower_bound_freq);
}


inline __device__ double GetZeroDopplerTime(double line_time_interval,
                                            double wavelength,
                                            snapengine::PosVector earth_point,
                                            snapengine::OrbitStateVectorComputation* orbit,
                                            const int num_orbit_vec,
                                            const double dt) {
    double first_vec_time = 0.0;
    double second_vec_time = 0.0;
    double first_vec_freq = 0.0;
    double second_vec_freq = 0.0;

    for (int i = 0; i < num_orbit_vec; i++) {
        snapengine::OrbitStateVectorComputation orb = orbit[i];

        double current_freq = getDopplerFrequency(earth_point, orb, wavelength);
        if (i == 0 || first_vec_freq * current_freq > 0) {
            first_vec_time = orb.timeMjd_;
            first_vec_freq = current_freq;
        } else {
            second_vec_time = orb.timeMjd_;
            second_vec_freq = current_freq;
            break;
        }
    }

    if (first_vec_freq * second_vec_freq >= 0.0) {
        return NON_VALID_ZERO_DOPPLER_TIME;
    }

    // find the exact time using Doppler frequency and bisection method
    double lower_bound_time = first_vec_time;
    double upper_bound_time = second_vec_time;
    double lower_bound_freq = first_vec_freq;
    double upper_bound_freq = second_vec_freq;
    double mid_time, mid_freq;
    double diff_time = std::abs(upper_bound_time - lower_bound_time);
    const double abs_line_time_interval = std::abs(line_time_interval);

    const int total_iterations = (int)(diff_time / abs_line_time_interval) + 1;
    int num_iterations = 0;
    while (diff_time > abs_line_time_interval && num_iterations <= total_iterations) {
        mid_time = (upper_bound_time + lower_bound_time) / 2.0;

        orbitstatevectors::PositionVelocity posvel =
            orbitstatevectors::GetPositionVelocity(mid_time, orbit, num_orbit_vec, dt);
        mid_freq = getDopplerFrequency(earth_point, posvel.position, posvel.velocity, wavelength);

        if (mid_freq * lower_bound_freq > 0.0) {
            lower_bound_time = mid_time;
            lower_bound_freq = mid_freq;
        } else if (mid_freq * upper_bound_freq > 0.0) {
            upper_bound_time = mid_time;
            upper_bound_freq = mid_freq;
            // TODO: there might be an accuracy bug here because we are missing a compare delta, but what is the delta?
        } else if (mid_freq == 0.0) {
            return mid_time;
        }

        diff_time = std::abs(upper_bound_time - lower_bound_time);
        num_iterations++;
    }

    return lower_bound_time - lower_bound_freq * (upper_bound_time - lower_bound_time) / (upper_bound_freq - lower_bound_freq);
}

inline __device__ __host__ double ComputeSlantRangeImpl(
    double time,
    cuda::KernelArray<snapengine::OrbitStateVectorComputation> vectors,
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
    if (range_index < 0.0 || range_index >= src_max_range || azimuth_index <= 0.0 || azimuth_index >= src_max_azimuth) {
        return false;
    }

    return diff_lat < 5;
}

}  // namespace sargeocoding
}  // namespace s1tbx
}  // namespace alus