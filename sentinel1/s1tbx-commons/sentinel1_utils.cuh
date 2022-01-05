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

#include "sentinel1_utils_computation.h"

#include "subswath_info_computation.h"

namespace alus {
namespace s1tbx {
__device__ inline Sentinel1Index ComputeIndex(double azimuth_time, double slant_range_time,
                                              DeviceSubswathInfo* subswath_info,
                                              const double* subswath_slant_range_times,
                                              const double* subswath_azimuth_times) {
    int j_0{-1};
    int j_1{-1};
    double mu_x{0.0};
    if (slant_range_time < subswath_slant_range_times[0]) {
        j_0 = 0;
        j_1 = 1;
    } else if (slant_range_time > subswath_slant_range_times[subswath_info->num_of_geo_points_per_line - 1]) {
        j_0 = subswath_info->num_of_geo_points_per_line - 2;
        j_1 = subswath_info->num_of_geo_points_per_line - 1;
    } else {
        for (int j = 0; j < subswath_info->num_of_geo_points_per_line - 1; j++) {
            if (subswath_slant_range_times[j] <= slant_range_time &&
                subswath_slant_range_times[j + 1] > slant_range_time) {
                j_0 = j;
                j_1 = j + 1;
                break;
            }
        }
    }
    mu_x = (slant_range_time - subswath_slant_range_times[j_0]) /
           (subswath_slant_range_times[j_1] - subswath_slant_range_times[j_0]);
    int i_0{-1};
    int i_1{-1};
    double mu_y{0.0};
    for (int i = 0; i < subswath_info->num_of_geo_lines - 1; i++) {
        int aux_index = i * subswath_info->num_of_geo_points_per_line;
        double i_0_azimuth_time =
            (1 - mu_x) * subswath_azimuth_times[aux_index + j_0] + mu_x * subswath_azimuth_times[aux_index + j_1];
        aux_index = (i + 1) * subswath_info->num_of_geo_points_per_line;
        double i_1_azimuth_time =
            (1 - mu_x) * subswath_azimuth_times[aux_index + j_0] + mu_x * subswath_azimuth_times[aux_index + j_1];
        if ((i == 0 && azimuth_time < i_0_azimuth_time) ||
            (i == subswath_info->num_of_geo_lines - 2 && azimuth_time >= i_1_azimuth_time) ||
            (i_0_azimuth_time <= azimuth_time && i_1_azimuth_time > azimuth_time)) {
            i_0 = i;
            i_1 = i + 1;
            mu_y = (azimuth_time - i_0_azimuth_time) / (i_1_azimuth_time - i_0_azimuth_time);
            break;
        }
    }
    return {i_0, i_1, j_0, j_1, mu_x, mu_y};
}

__device__ inline double GetLatitude(Sentinel1Index& index, double const* latitudes, int number_of_pixels_per_line) {
    double lat_00 = latitudes[index.i0 * number_of_pixels_per_line + index.j0];
    double lat_01 = latitudes[index.i0 * number_of_pixels_per_line + index.j1];
    double lat_10 = latitudes[index.i1 * number_of_pixels_per_line + index.j0];
    double lat_11 = latitudes[index.i1 * number_of_pixels_per_line + index.j1];
    return (1 - index.mu_y) * ((1 - index.mu_x) * lat_00 + index.mu_x * lat_01) +
           index.mu_y * ((1 - index.mu_x) * lat_10 + index.mu_x * lat_11);
}

__device__ inline double GetLongitude(Sentinel1Index& index, double const* longitudes, int number_of_pixels_per_line) {
    double lon_00 = longitudes[index.i0 * number_of_pixels_per_line + index.j0];
    double lon_01 = longitudes[index.i0 * number_of_pixels_per_line + index.j1];
    double lon_10 = longitudes[index.i1 * number_of_pixels_per_line + index.j0];
    double lon_11 = longitudes[index.i1 * number_of_pixels_per_line + index.j1];
    return (1 - index.mu_y) * ((1 - index.mu_x) * lon_00 + index.mu_x * lon_01) +
           index.mu_y * ((1 - index.mu_x) * lon_10 + index.mu_x * lon_11);
}

}  // namespace s1tbx
}  // namespace alus