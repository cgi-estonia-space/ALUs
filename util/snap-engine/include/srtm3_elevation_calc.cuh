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

inline __device__ int GetSamples(
    PointerArray *tiles, int *x, int *y, double *samples, int width, int height, double no_value, int use_no_data) {
    // in this case, it will always be used.
    no_value = no_value;
    int allValid = 1;
    int i = 0, j = 0;
    int tile_y_index, tile_x_index, pixel_y, pixel_x;
    int xI, yI;
    float *srtm_41_01_tile = (float *)tiles->array[0].pointer;
    int xSize = tiles->array[0].x;
    float *srtm_42_01_tile = (float *)tiles->array[1].pointer;

    for (yI = 0; yI < height; yI++) {
        tile_y_index = (int)(y[yI] * NUM_PIXELS_PER_TILEinv);
        pixel_y = y[yI] - tile_y_index * NUM_PIXELS_PER_TILE;

        j = 0;
        for (xI = 0; xI < width; xI++) {
            tile_x_index = (int)(x[xI] * NUM_PIXELS_PER_TILEinv);

            // make sure that the tile we want is actually listed
            if (tile_x_index > NUM_X_TILES || tile_x_index < 0 || tile_y_index > NUM_Y_TILES || tile_y_index < 0) {
                samples[i * width + j] = CUDART_NAN;
                allValid = 0;
                ++j;
                continue;
            }
            pixel_x = x[xI] - tile_x_index * NUM_PIXELS_PER_TILE;

            // TODO: placeholder. Change once you know how dynamic tiling will work.
            switch (tile_x_index) {
                case 40:
                    samples[i * width + j] = srtm_41_01_tile[pixel_x + xSize * pixel_y];
                    break;
                case 41:
                    samples[i * width + j] = srtm_42_01_tile[pixel_x + xSize * pixel_y];
                    break;
                default:
                    printf("Slave pix pos where it should not be. %d \n", tile_x_index);
                    samples[i * width + j] = CUDART_NAN;
            }

            if (samples[i * width + j] == NO_DATA_VALUE) {
                samples[i * width + j] = CUDART_NAN;
                allValid = 0;
            }
            ++j;
        }
        ++i;
    }
    return allValid;
}

inline __device__ double GetElevation(double geo_pos_lat, double geo_pos_lon, PointerArray *p_array) {
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
