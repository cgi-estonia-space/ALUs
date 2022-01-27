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

#include <cstddef>

namespace alus {  // NOLINT TODO: concatenate namespace and remove nolint after migrating to cuda 11+
namespace s1tbx {
struct DeviceSubswathInfo {
    // subswath info
    int first_valid_pixel;
    int last_valid_pixel;
    double first_line_time;
    double last_line_time;
    double slr_time_to_first_pixel;
    double slr_time_to_last_pixel;
    double range_pixel_spacing;
    double azimuth_time_interval;
    double radar_frequency;
    double azimuth_steering_rate;

    double* device_burst_first_line_time;
    double* device_burst_last_line_time;
    size_t burst_line_times_count;

    double* device_doppler_rate;
    double* device_doppler_centroid;
    double* device_reference_time;
    double* device_range_depend_doppler_rate;
    int doppler_size_x, doppler_size_y;

    double* device_subswath_azimuth_times;
    double* device_subswath_slant_range_times;
    double* device_latidudes;
    double* device_longitudes;
    // GeoLocationGridPoint
    int num_of_geo_lines;
    int num_of_geo_points_per_line;

    // bursts info
    int lines_per_burst;
    int num_of_bursts;
    int samples_per_burst;
};
}  // namespace s1tbx
}  // namespace alus
