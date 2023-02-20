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

#include <math_constants.h>

#include "dem_property.h"
#include "resampling.h"
#include "snap-core/core/dataop/resamp/bilinear_interpolation.cuh"

namespace alus::dem {

inline __device__ int GetSamples(PointerArray* tiles, int* x, int* y, double* samples, int width, int height,
                                 double /*no_value*/, int /*use_no_data*/, const alus::dem::Property* dem_prop) {
    int all_valid = 1;
    int tile_pixel_count_x = dem_prop[0].tile_pixel_count_x;

    int i = 0;
    for (int yI = 0; yI < height; yI++) {
        const int tile_y_index = (int)(y[yI] * dem_prop->tile_pixel_count_inverted_y);
        const int pixel_y = y[yI] - tile_y_index * dem_prop->tile_pixel_count_y;

        int j = 0;
        for (int xI = 0; xI < width; xI++) {
            const int tile_x_index = (int)(x[xI] * dem_prop->tile_pixel_count_inverted_x);

            const int samples_index = i * width + j;
            // make sure that the tile we want is actually listed
            if (tile_x_index > static_cast<int>(dem_prop->grid_tile_count_x) || tile_x_index < 0 ||
                tile_y_index > static_cast<int>(dem_prop->grid_tile_count_y) || tile_y_index < 0) {
                samples[samples_index] = CUDART_NAN;
                all_valid = 0;
                ++j;
                continue;
            }
            const int pixel_x = x[xI] - tile_x_index * dem_prop->tile_pixel_count_x;
            const int tile_pixel_index = pixel_x + tile_pixel_count_x * pixel_y;
            const int tile_id = tile_x_index * 1000 + tile_y_index + 1;
            for (int tile_i = 0; tile_i < (int)tiles->size; tile_i++) {
                if (tiles->array[tile_i].id == tile_id) {
                    const float* array = (float*)tiles->array[tile_i].pointer;
                    const float value = array[tile_pixel_index];
//                    printf("tile pixel index %d value %f\n", tile_pixel_index, value);
                    samples[samples_index] = value;
                    break;
                }
            }

            if (samples[samples_index] == dem_prop->no_data_value) {
                samples[samples_index] = CUDART_NAN;
                all_valid = 0;
            }
            ++j;
        }
        ++i;
    }
    return all_valid;
}

inline __device__ __host__ size_t GetCopDemCog30mTileWidth(double lat) {
    const int abs_lat = (int)abs(lat);
    if (abs_lat < 50) {
        return 3600;
    } else if (abs_lat >= 50 && abs_lat < 60) {
        return 2400;
    } else if (abs_lat >= 60 && abs_lat < 70) {
        return 1800;
    } else if (abs_lat >= 70 && abs_lat < 80) {
        return 1200;
    } else if (abs_lat >= 80 && abs_lat < 85) {
        return 720;
    } else {
        return 360;
    }
}

inline __device__ __host__ const Property* GetCopDemPropertyBy(double lat, const Property* dem_props, size_t count) {
    const auto tile_width_for_lat = GetCopDemCog30mTileWidth(lat);
    for (size_t i = 0; i < count; i++) {
        if (dem_props[i].tile_pixel_count_x == tile_width_for_lat) {
            return dem_props + i;
        }
    }

    return nullptr;
}

inline __device__ double CopDemCog30mGetElevation(double geo_pos_lat, double geo_pos_lon, PointerArray* p_array,
                                                  const alus::dem::Property* dem_prop) {
    if (geo_pos_lon > 180.0) {
        geo_pos_lon -= 360.0;
    }

    const Property* dp = GetCopDemPropertyBy(geo_pos_lat, dem_prop, p_array->size);

    if (dp == nullptr) {
        return dem_prop->no_data_value;
    }

    double pixel_y = (dp->grid_max_lat - geo_pos_lat) * dp->tile_pixel_size_deg_inverted_y;
    if (pixel_y < 0 || isnan(pixel_y)) {
        return dp->no_data_value;
    }
    double pixel_x = (geo_pos_lon + dp->grid_max_lon) * dp->tile_pixel_size_deg_inverted_x;
//    printf("pixel x %f y %f\n", pixel_x, pixel_y);
    // computing corner based index
    double index_i[2];
    double index_j[2];
    double index_ki[1];
    double index_kj[1];
    snapengine::resampling::ResamplingIndex index{0, 0, 0, 0, 0, 0, index_i, index_j, index_ki, index_kj};
    snapengine::bilinearinterpolation::ComputeIndex(pixel_x + 0.5, pixel_y + 0.5,
                                                    static_cast<int>(dp->grid_total_width_pixels),
                                                    static_cast<int>(dp->grid_total_height_pixels), &index);

    auto elevation =
        snapengine::bilinearinterpolation::Resample(p_array, &index, 2, CUDART_NAN, 1, dp, GetSamples);
//    printf("elev %f\n", elevation);
    return isnan(elevation) ? dp->no_data_value : elevation;
}

}  // namespace alus::dem
