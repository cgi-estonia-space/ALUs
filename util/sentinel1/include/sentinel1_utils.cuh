#pragma once

#include <cuda_runtime.h>

#include "subswath_info.cuh"

namespace alus {
namespace s1tbx {

struct DeviceSentinel1Utils {
    double first_line_utc;
    double last_line_utc;
    double line_time_interval;
    double near_edge_slant_range;
    double wavelength;
    double range_spacing;
    double azimuth_spacing;

    int source_image_width;
    int source_image_height;
    int near_range_on_left;
    int srgr_flag;
};

struct Sentinel1Index {
    int i0;
    int i1;
    int j0;
    int j1;
    double mu_x;
    double mu_y;
};

__device__ inline Sentinel1Index ComputeIndex(double azimuth_time,
                                                  double slant_range_time,
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

__device__ inline double GetLatitude(Sentinel1Index& index,
                                         double const* latitudes,
                                         int number_of_pixels_per_line) {
    double lat_00 = latitudes[index.i0 * number_of_pixels_per_line + index.j0];
    double lat_01 = latitudes[index.i0 * number_of_pixels_per_line + index.j1];
    double lat_10 = latitudes[index.i1 * number_of_pixels_per_line + index.j0];
    double lat_11 = latitudes[index.i1 * number_of_pixels_per_line + index.j1];

    return (1 - index.mu_y) * ((1 - index.mu_x) * lat_00 + index.mu_x * lat_01) +
           index.mu_y * ((1 - index.mu_x) * lat_10 + index.mu_x * lat_11);
}

__device__ inline double GetLongitude(Sentinel1Index& index,
                                          double const* longitudes,
                                          int number_of_pixels_per_line) {
    double lon_00 = longitudes[index.i0 * number_of_pixels_per_line + index.j0];
    double lon_01 = longitudes[index.i0 * number_of_pixels_per_line + index.j1];
    double lon_10 = longitudes[index.i1 * number_of_pixels_per_line + index.j0];
    double lon_11 = longitudes[index.i1 * number_of_pixels_per_line + index.j1];

    return (1 - index.mu_y) * ((1 - index.mu_x) * lon_00 + index.mu_x * lon_01) +
           index.mu_y * ((1 - index.mu_x) * lon_10 + index.mu_x * lon_11);
}

}  // namespace s1tbx
}  // namespace alus