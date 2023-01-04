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

#include <vector>

#include <boost/date_time/posix_time/ptime.hpp>

namespace alus::palsar {
struct ChirpInfo {
    double range_sampling_rate;
    double pulse_duration;
    int n_samples;
    double pulse_bandwidth;
    double coefficient[5];
};

struct OrbitInfo {
    double time_point;
    double x_pos;
    double y_pos;
    double z_pos;
    double x_vel;
    double y_vel;
    double z_vel;

    void Print() const
    {
        printf("OSV=%f,%f,%f,%f,%f,%f,%f\n", time_point, x_pos, y_pos, z_pos, x_vel, y_vel, z_vel);
    }
};

struct ImgFormat {
    int range_size;
    int azimuth_size;
    int data_line_offset;
    int record_length;
};

struct SARResults {
    size_t total_samples;
    double dc_i;
    double dc_q;
    double Vr;
    double doppler_centroid;
};

struct SARMetadata {
    ChirpInfo chirp;
    ImgFormat img;
    std::string polarisation;
    std::vector<uint32_t> slant_range_times;
    std::vector<uint32_t> left_range_offsets;
    boost::posix_time::ptime first_orbit_time;
    double orbit_interval;
    std::vector<OrbitInfo> orbit_state_vectors;
    double pulse_repetition_frequency;
    double azimuth_bandwidth_fraction;
    double carrier_frequency;
    double wavelength;
    OrbitInfo first_position;
    double platform_velocity;
    double range_spacing;
    double azimuth_spacing;
    double slant_range_first_sample;
    double center_lat;
    double center_lon;
    boost::posix_time::ptime center_time;

    SARResults results;
};

}  // namespace alus::palsar
