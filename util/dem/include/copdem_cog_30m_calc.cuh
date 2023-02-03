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

#include <stdio.h>
#include <math_constants.h>

#include "dem_calc.h"
#include "dem_property.h"
#include "resampling.h"
#include "snap-core/core/dataop/resamp/bilinear_interpolation.cuh"

namespace alus::dem {

inline __device__ int GetSamples(PointerArray* tiles, int* x, int* y, double* samples, int width, int height,
                                 double /*no_value*/, int /*use_no_data*/, const alus::dem::Property* dem_prop) {
    int all_valid = 1;
    int tile_pixel_count_x = dem_prop[0].pixels_per_tile_x_axis;

    int i = 0;
    for (int yI = 0; yI < height; yI++) {
        const int tile_y_index = (int)(y[yI] * dem_prop->pixels_per_tile_inverted_y_axis);
        const int pixel_y = y[yI] - tile_y_index * dem_prop->pixels_per_tile_y_axis;

        int j = 0;
        for (int xI = 0; xI < width; xI++) {
            const int tile_x_index = (int)(x[xI] * dem_prop->pixels_per_tile_inverted_x_axis);

            const int samples_index = i * width + j;
            // make sure that the tile we want is actually listed
            if (tile_x_index > static_cast<int>(dem_prop->tiles_x_axis) || tile_x_index < 0 ||
                tile_y_index > static_cast<int>(dem_prop->tiles_y_axis) || tile_y_index < 0) {
                samples[samples_index] = CUDART_NAN;
                all_valid = 0;
                ++j;
                continue;
            }
            const int pixel_x = x[xI] - tile_x_index * dem_prop->pixels_per_tile_x_axis;
            const int tile_pixel_index = pixel_x + tile_pixel_count_x * pixel_y;
            const int tile_id = tile_x_index * 1000 + tile_y_index;
            printf("ID %d x_i %d y_i %d\n", tile_id, tile_x_index, tile_y_index);
            for (int tile_i = 0; tile_i < (int)tiles->size; tile_i++) {
                if (tiles->array[tile_i].id == tile_id) {
                    const float* array = (float*)tiles->array[tile_i].pointer;
                    const float value = array[tile_pixel_index];
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

inline __device__ __host__ size_t GetCopDemCog30mRasterWidth(double lat) {
    const int abs_lat = (int)abs(lat);
    if (abs_lat < 50) {
        return 1200;
    } else if (abs_lat >= 50 && abs_lat < 60) {
        return 800;
    } else if (abs_lat >= 60 && abs_lat < 70) {
        return 600;
    } else if (abs_lat >= 70 && abs_lat < 75) {
        return 400;
    } else if (abs_lat >= 75 && abs_lat < 80) {
        return 400;
    } else if (abs_lat >= 80 && abs_lat < 85) {
        return 240;
    } else {
        return 120;
    }
}

inline __device__ double CopDemCog30mGetElevation(double geo_pos_lat, double geo_pos_lon, PointerArray* p_array,
                                                  const alus::dem::Property* dem_prop) {
    if (geo_pos_lon > 180.0) {
        geo_pos_lon -= 360.0;
    }

    const auto raster_width_for_pos = GetCopDemCog30mRasterWidth(geo_pos_lat);
    const auto dem_tile_count = p_array->size;
    const alus::dem::Property* dp = nullptr;
    for (size_t i = 0; i < dem_tile_count; i++) {
        if (dem_prop[i].raster_width == raster_width_for_pos) {
            dp = dem_prop + i;
            break;
        }
    }

    if (dp == nullptr) {
        return dem_prop->no_data_value;
    }

    double pixel_y = (dp->lat_coverage - geo_pos_lat) * dp->pixel_size_degrees_inverted_y_axis;
    if (pixel_y < 0 || isnan(pixel_y)) {
        return dp->no_data_value;
    }
    double pixel_x = (geo_pos_lon + dp->lon_coverage) * dp->pixel_size_degrees_inverted_x_axis;

    // computing corner based index
    double index_i[2];
    double index_j[2];
    double index_ki[1];
    double index_kj[1];
    snapengine::resampling::ResamplingIndex index{0, 0, 0, 0, 0, 0, index_i, index_j, index_ki, index_kj};
    snapengine::bilinearinterpolation::ComputeIndex(pixel_x + 0.5, pixel_y + 0.5,
                                                    static_cast<int>(dp->raster_width),
                                                    static_cast<int>(dp->raster_height), &index);

    auto elevation =
        snapengine::bilinearinterpolation::Resample(p_array, &index, 2, CUDART_NAN, 1, dp, GetSamples);

    return isnan(elevation) ? dp->no_data_value : elevation;
}

__device__ alus::dem::GetElevationFunc get_elevation_cop_dem_cog_30m = alus::dem::CopDemCog30mGetElevation;

}  // namespace alus::dem