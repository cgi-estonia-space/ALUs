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

#include <algorithm>

#include "pointer_holders.h"
#include "resampling.h"
#include "shapes.h"

#include "bilinear_interpolation.cuh"
#include "cuda_util.cuh"
#include "resampling.cuh"

namespace alus {
namespace terraincorrection {
namespace rangedopplergeocoding {

inline __device__ __host__ Rectangle ComputeSourceRectangle(double range_index, double azimuth_index, int margin,
                                                                  int source_image_width, int source_image_height) {
    int x = std::max(static_cast<int>(range_index) - margin, 0);
    int y = std::max(static_cast<int>(azimuth_index) - margin, 0);
    int x_max = std::min(x + 2 * margin + 1, source_image_width);
    int y_max = std::min(y + 2 * margin + 1, source_image_height);
    int width = x_max - x;
    int height = y_max - y;

    return Rectangle{x, y, width, height};
}

/**
 * Utility method, that gets the subimage of the given image
 *
 * @param src_tile source image represented by the Tile class
 * @param rectangle desired subimage size and coordinates
 */
inline __device__ __host__ void GetSubTileData(const snapengine::resampling::Tile& src_tile,const Rectangle& rectangle,
                                               double* dest_data_buffer) {
    int offset = 0;
    for (int y = rectangle.y; y < rectangle.y + rectangle.height; y++) {
        for (int x = rectangle.x; x < rectangle.x + rectangle.width; x++) {
            auto index = y * src_tile.width + x;
            if (index > src_tile.width * src_tile.height) {
                continue;
            }
            dest_data_buffer[offset++] = src_tile.data_buffer[index];
        }
    }
}

inline __device__ double Resample(PointerArray* tiles, snapengine::resampling::ResamplingIndex* index, int raster_width,
                                  double no_value, int use_no_data,
                                  int get_samples_function(PointerArray*, int*, int*, double*, int, int, double, int)) {
    return snapengine::bilinearinterpolation::Resample(tiles, index, raster_width, no_value, use_no_data,
                                                       get_samples_function);
}

inline __device__ int GetSamples(PointerArray* tiles, int* x_values, int* y_values, double* samples, int width,
                                 int height, double no_value, int use_no_data) {
    bool all_valid = true;
    auto* values = (float*)tiles->array[0].pointer;

    for (int y = 0; y < 2; y++) {
        for (int x = 0; x < 2; x++) {
            auto index = y_values[y] * width + x_values[x];
            samples[y * 2 + x] = values[index];
        }
    }

    return all_valid;
}

inline __device__ __host__ void AdjustResamplingIndex(snapengine::resampling::ResamplingIndex& index,
                                                      const snapengine::resampling::ResamplingRaster& raster) {
    index.x -= raster.source_tile_i->x_0;
    index.y -= raster.source_tile_i->y_0;
    index.i0 -= raster.source_tile_i->x_0;
    index.j0 -= raster.source_tile_i->y_0;
    index.i[0] -= raster.source_tile_i->x_0;
    index.i[1] -= raster.source_tile_i->x_0;
    index.j[0] -= raster.source_tile_i->y_0;
    index.j[1] -= raster.source_tile_i->y_0;
}

inline __device__ double GetPixelValue(double azimuth_index, double range_index, int margin, int source_image_width,
                                       int source_image_height,
                                       snapengine::resampling::ResamplingRaster resampling_raster,
                                       snapengine::resampling::ResamplingIndex resampling_index, int& sub_swath_index) {

    snapengine::resampling::SetRangeAzimuthIndicesImpl(resampling_raster, range_index, azimuth_index);

    snapengine::bilinearinterpolation::ComputeIndex(range_index + 0.5, azimuth_index + 0.5, source_image_width,
                                                    source_image_height, &resampling_index);

    AdjustResamplingIndex(resampling_index, resampling_raster);

    PointerArray p_array;
    PointerHolder p_holder;
    p_array.array = &p_holder;
    p_holder.x = source_image_width;
    p_holder.y = source_image_height;
    p_holder.pointer = resampling_raster.source_tile_i->data_buffer;
    double v = Resample(&p_array, &resampling_index, resampling_raster.source_tile_i->width, 0, 0, GetSamples); // TODO(anton): change correctness

    sub_swath_index = resampling_raster.sub_swath_index;

    sub_swath_index = resampling_raster.sub_swath_index;

    return v;
}
}  // namespace rangedopplergeocoding
}  // namespace terraincorrection
}  // namespace alus