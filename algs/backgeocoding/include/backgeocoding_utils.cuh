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

#include "burst_indices_computation.h"
#include "general_constants.h"
#include "position_data.h"
#include "snap-engine-utilities/engine-utilities/eo/constants.h"

#include "s1tbx-commons/sar_geocoding.cuh"
#include "s1tbx-commons/sentinel1_utils.cuh"

namespace alus {
namespace backgeocoding {
__device__ __host__ inline int GetBurstIndex(int y, int number_of_bursts, int lines_per_burst) {
    int index = y / lines_per_burst;
    return (index >= number_of_bursts) * -1 + (index < number_of_bursts) * index;
}

__device__ __host__ inline double GetAzimuthTime(int y, int burst_index, s1tbx::DeviceSubswathInfo* subswath_info) {
    return subswath_info->device_burst_first_line_time[burst_index] +
           (y - burst_index * subswath_info->lines_per_burst) * subswath_info->azimuth_time_interval;
}

__device__ __host__ inline double GetSlantRangeTime(int x, s1tbx::DeviceSubswathInfo* subswath_info) {
    return subswath_info->slr_time_to_first_pixel +
           x * subswath_info->range_pixel_spacing / snapengine::eo::constants::LIGHT_SPEED;
}

inline __device__ bool GetPosition(s1tbx::DeviceSubswathInfo* subswath_info,
                                   s1tbx::DeviceSentinel1Utils* sentinel1_utils, int burst_index,
                                   s1tbx::PositionData* position_data, snapengine::OrbitStateVectorComputation* orbit,
                                   const int num_orbit_vec, const double dt, int idx, int idy) {
    const double zero_doppler_time_in_days =
        s1tbx::sargeocoding::GetZeroDopplerTime(sentinel1_utils->line_time_interval, sentinel1_utils->wavelength,
                                                position_data->earth_point, orbit, num_orbit_vec, dt);

    if (zero_doppler_time_in_days == s1tbx::sargeocoding::NON_VALID_ZERO_DOPPLER_TIME) {
        return false;
    }

    const double zero_doppler_time = zero_doppler_time_in_days * snapengine::eo::constants::SECONDS_IN_DAY;
    position_data->azimuth_index = burst_index * subswath_info->lines_per_burst +
                                   (zero_doppler_time - subswath_info->device_burst_first_line_time[burst_index]) /
                                       subswath_info->azimuth_time_interval;

    // this is here to force a very specific compilation. This changes values of the above operation closer to snap
    // TODO: consider getting rid of it once the whole algorithm is complete and we no longer need snap to debug.
    // https://jira.devzone.ee/browse/SNAPGPU-169
    if (idx == 55 && idy == 53) {
        //printf("AZ index %.10f zero doppler time: %.10f FLT %.10f AZTI %.10f zero doppler in days %.10f\n",
        //       position_data->azimuth_index, zero_doppler_time,
        //       subswath_info->device_burst_first_line_time[burst_index], subswath_info->azimuth_time_interval,
        //       zero_doppler_time_in_days);
        if(isnan(zero_doppler_time)) { //workaround to get rid of the previous printf
            return false;
        }
    }

    cuda::KernelArray<snapengine::OrbitStateVectorComputation> orbit_vectors;

    orbit_vectors.array = orbit;
    orbit_vectors.size = num_orbit_vec;
    const double SLANT_RANGE = s1tbx::sargeocoding::ComputeSlantRangeImpl(
        zero_doppler_time_in_days, orbit_vectors, position_data->earth_point, position_data->sensor_pos);

    if (!sentinel1_utils->srgr_flag) {
        position_data->range_index =
            (SLANT_RANGE - subswath_info->slr_time_to_first_pixel * snapengine::eo::constants::LIGHT_SPEED) /
            sentinel1_utils->range_spacing;
    } else {
        // TODO: implement this some day, as we don't need it for first demo.
        /*position_data->range_index = s1tbx::sargeocoding::computeRangeIndex(
            su.srgrFlag, su.sourceImageWidth, su.firstLineUTC, su.lastLineUTC,
            su.rangeSpacing, zeroDopplerTimeInDays, slantRange, su.nearEdgeSlantRange, su.srgrConvParams);*/
    }

    if (!sentinel1_utils->near_range_on_left) {
        position_data->range_index = sentinel1_utils->source_image_width - 1 - position_data->range_index;
    }

    return true;
}

__device__ inline BurstIndices GetBurstIndices(double line_time_interval, double wavelength, int num_of_bursts,
                                               const double* burst_first_line_times,
                                               const double* burst_last_line_times, snapengine::PosVector earth_point,
                                               snapengine::OrbitStateVectorComputation* orbit,
                                               const size_t num_orbit_vec, const double dt) {
    BurstIndices burst_indices{-1, -1, false, false, false};
    const double zero_doppler_time_in_days =
        s1tbx::sargeocoding::GetZeroDopplerTime(line_time_interval, wavelength, earth_point, orbit, num_orbit_vec, dt);
    if (zero_doppler_time_in_days == s1tbx::sargeocoding::NON_VALID_ZERO_DOPPLER_TIME) {
        return burst_indices;
    }

    const double zero_doppler_time = zero_doppler_time_in_days * snapengine::eo::constants::SECONDS_IN_DAY;
    for (int i = 0, k = 0; i < num_of_bursts; i++) {
        if (zero_doppler_time >= burst_first_line_times[i] && zero_doppler_time < burst_last_line_times[i]) {
            bool in_upper_part = (zero_doppler_time >= (burst_first_line_times[i] + burst_last_line_times[i]) / 2.0);
            if (k == 0) {
                burst_indices.first_burst_index = i;
                burst_indices.in_upper_part_of_first_burst = in_upper_part;
            } else {
                burst_indices.second_burst_index = i;
                burst_indices.in_upper_part_of_second_burst = in_upper_part;
                break;
            }
            k++;
        }
    }
    burst_indices.valid = (burst_indices.first_burst_index != -1) || (burst_indices.second_burst_index != -1);
    return burst_indices;
}

__device__ inline BurstIndices GetBurstIndices(const s1tbx::DeviceSentinel1Utils* sentinel_utils,
                                               const s1tbx::DeviceSubswathInfo* subswath_info,
                                               snapengine::PosVector earth_point,
                                               snapengine::OrbitStateVectorComputation* orbit,
                                               const size_t num_orbit_vec, const double dt) {
    return GetBurstIndices(sentinel_utils->line_time_interval, sentinel_utils->wavelength, subswath_info->num_of_bursts,
                           subswath_info->device_burst_first_line_time, subswath_info->device_burst_last_line_time,
                           earth_point, orbit, num_orbit_vec, dt);
}
}  // namespace backgeocoding
}  // namespace alus