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

#include <cmath>

#include "bilinear_interpolation.cuh"
#include "cuda_util.cuh"
#include "pointer_holders.h"
#include "resampling.cuh"
#include "shapes.h"

namespace alus {
namespace terraincorrection {
namespace rangedopplergeocoding {

inline __device__ __host__ alus::Rectangle ComputeSourceRectangle(
    double range_index, double azimuth_index, int margin, int source_image_width, int source_image_height) {
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
inline __device__ __host__ void GetSubTileData(const alus::snapengine::resampling::Tile &src_tile,
                                               alus::Rectangle &rectangle,
                                               double *dest_data_buffer) {
    int offset = 0;
    for (int y = rectangle.y; y < rectangle.y + rectangle.height; y++) {
        for (int x = rectangle.x; x < rectangle.x + rectangle.width; x++) {
            long index = static_cast<long>(y) * src_tile.width + x;
            if (index > static_cast<long>(src_tile.width) * src_tile.height) {
                continue;
            }
            dest_data_buffer[offset++] = src_tile.data_buffer[index];
        }
    }
}

inline __device__ double Resample(
    PointerArray *tiles,
    snapengine::resampling::ResamplingIndex *index,
    int raster_width,
    double no_value,
    int use_no_data,
    int GetSamplesFunction(PointerArray *, int *, int *, double *, int, int, double, int)) {
    return snapengine::bilinearinterpolation::Resample(
        tiles, index, raster_width, no_value, use_no_data, GetSamplesFunction);
}

inline __device__ int GetSamples(
    PointerArray *tiles, int *x, int *y, double *samples, int width, int height, double no_value, int use_no_data) {
    bool all_valid = true;
    auto *values = (double *)tiles->array[0].pointer;

    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            int y_adjusted = i + 1;
            int x_adjusted = j + 1;
            int index = width * y_adjusted + x_adjusted;
            double v = values[index];

            samples[i * 2 + j] = v;
        }
    }

    return all_valid;
}

inline __device__ double GetPixelValue(double azimuth_index,
                                       double range_index,
                                       int margin,
                                       int source_image_width,
                                       int source_image_height,
                                       alus::snapengine::resampling::TileData *tile_data,
                                       const double *band_data_buffer,  // TODO:  unnecessary?
                                       int &sub_swath_index) {
//    double *dest_data_buffer;
    bool compute_new_source_rectangle = false;
//
//    source_image_width = tile_data->source_tile->width;
//    source_image_height = tile_data->source_tile->height;

    if (tile_data->resampling_raster->source_rectangle) {
        alus::Rectangle &rectangle = *tile_data->resampling_raster->source_rectangle;
        const int x_min = rectangle.x + margin;
        const int y_min = rectangle.y + margin;
        const int x_max = x_min + rectangle.width - 1 - 2 * margin;
        const int y_max = y_min + rectangle.height - 1 - 2 * margin;
        if (range_index < x_min || range_index > x_max || azimuth_index < y_min || azimuth_index > y_max) {
            compute_new_source_rectangle = true;
        }
    } else {
        compute_new_source_rectangle = true;
    }
    if (compute_new_source_rectangle) {
        return 0.0;
        // TODO: the code is currently commented out until GetSourceRectangle is correctly implemented (SNAPGPU-163)
//        alus::Rectangle src_rectangle =
//            ComputeSourceRectangle(range_index, azimuth_index, margin, source_image_width, source_image_height);
//        alus::snapengine::resampling::SetSourceRectangleImpl(*tile_data->resampling_raster, src_rectangle);
//
//        tile_data->source_tile->data_buffer = const_cast<double *>(band_data_buffer);
//
//        dest_data_buffer = new double[src_rectangle.width * src_rectangle.height];
//        GetSubTileData(
//            *tile_data->source_tile, src_rectangle, dest_data_buffer);
//
//        alus::snapengine::resampling::AssignTileValuesImpl(tile_data->resampling_raster->source_tile_i,
//                                                           src_rectangle.width,
//                                                           src_rectangle.height,
//                                                           tile_data->source_tile->target,
//                                                           tile_data->source_tile->scaled,
//                                                           dest_data_buffer);
    }

    alus::snapengine::resampling::SetRangeAzimuthIndicesImpl(tile_data->resampling_raster, range_index, azimuth_index);

    snapengine::bilinearinterpolation::ComputeIndex(range_index + 0.5,
                                                    azimuth_index + 0.5,
                                                    source_image_width,
                                                    source_image_height,
                                                    tile_data->image_resampling_index);

    PointerArray p_array;
    PointerHolder p_holder;
    p_array.array = &p_holder;
    p_holder.x = source_image_width;
    p_holder.y = source_image_height;
    p_holder.pointer = tile_data->resampling_raster->source_tile_i->data_buffer;

    double v = Resample(&p_array,
                        tile_data->image_resampling_index,
                        tile_data->resampling_raster->source_tile_i
                            ->width,  // TODO: width should probably be equal to 2 -> this is samples width
                        0,
                        0,
                        GetSamples);

    sub_swath_index = tile_data->resampling_raster->sub_swath_index;

    if (compute_new_source_rectangle) {
        tile_data->resampling_raster->source_tile_i = tile_data->source_tile;
//        delete[] dest_data_buffer;
    }

    return v;
}
}  // namespace rangedopplergeocoding
}  // namespace terraincorrection
}  // namespace alus