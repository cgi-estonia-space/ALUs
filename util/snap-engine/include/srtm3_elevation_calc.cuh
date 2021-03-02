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

#include <cstdio>

#include "bilinear_interpolation.cuh"
#include "math_constants.h"  //not sure if required
#include "resampling.h"
#include "srtm3_elevation_model_constants.h"

namespace alus {
namespace snapengine {
namespace srtm3elevationmodel {

inline __device__ int GetSamples(PointerArray* tiles, int* x, int* y, double* samples, int width, int height,
                                 double /*no_value*/, int /*use_no_data*/) {
    int allValid = 1;
    int xSize = tiles->array[0].x;

    int i = 0;
    for (int yI = 0; yI < height; yI++) {
        const int tile_y_index = (int)(y[yI] * NUM_PIXELS_PER_TILEinv);
        const int pixel_y = y[yI] - tile_y_index * NUM_PIXELS_PER_TILE;

        int j = 0;
        for (int xI = 0; xI < width; xI++) {
            const int tile_x_index = (int)(x[xI] * NUM_PIXELS_PER_TILEinv);

            const int samples_index = i * width + j;
            // make sure that the tile we want is actually listed
            if (tile_x_index > NUM_X_TILES || tile_x_index < 0 || tile_y_index > NUM_Y_TILES || tile_y_index < 0) {
                samples[samples_index] = CUDART_NAN;
                allValid = 0;
                ++j;
                continue;
            }
            const int pixel_x = x[xI] - tile_x_index * NUM_PIXELS_PER_TILE;
            const int srtm_source_index = pixel_x + xSize * pixel_y;
            // ID field in PointerHolder is used for identifying SRTM3 tile indexes. File data in srtm_42_01.tif results
            // in ID with 4201. Yes, it is hacky.
            const int srtm_id = (tile_x_index + 1) * 100 + (tile_y_index + 1);
            for (int srtm_tiles_i = 0; srtm_tiles_i < (int)tiles->size; srtm_tiles_i++) {
                if (tiles->array[srtm_tiles_i].id == srtm_id) {
                    const float* array = (float*)tiles->array[srtm_tiles_i].pointer;
                    const float value = array[srtm_source_index];
                    samples[samples_index] = value;
                    break;
                }
            }

            if (samples[samples_index] == NO_DATA_VALUE) {
                samples[samples_index] = CUDART_NAN;
                allValid = 0;
            }
            ++j;
        }
        ++i;
    }
    return allValid;
}

inline __device__ double GetElevation(double geo_pos_lat, double geo_pos_lon, PointerArray* p_array) {
    double index_i[2];
    double index_j[2];
    double index_ki[1];
    double index_kj[1];
    snapengine::resampling::ResamplingIndex index{0, 0, 0, 0, 0, 0, index_i, index_j, index_ki, index_kj};

    if (geo_pos_lon > 180.0) {
        geo_pos_lat -= 360.0;
    }

    double pixel_y = (60.0 - geo_pos_lat) * DEGREE_RES_BY_NUM_PIXELS_PER_TILEinv;
    if (pixel_y < 0 || isnan(pixel_y)) {
        return NO_DATA_VALUE;
    }
    double pixel_x = (geo_pos_lon + 180.0) * DEGREE_RES_BY_NUM_PIXELS_PER_TILEinv;
    double elevation = 0.0;

    // computing corner based index.
    snapengine::bilinearinterpolation::ComputeIndex(pixel_x + 0.5, pixel_y + 0.5, RASTER_WIDTH, RASTER_HEIGHT, &index);

    elevation = snapengine::bilinearinterpolation::Resample(p_array, &index, 2, CUDART_NAN, 1, GetSamples);

    return isnan(elevation) ? NO_DATA_VALUE : elevation;
}

}  // namespace srtm3elevationmodel
}  // namespace snapengine
}  // namespace alus
