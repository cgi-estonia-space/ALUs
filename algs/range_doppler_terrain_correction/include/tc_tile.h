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

#include <cuda_util.cuh>

namespace alus {

struct TcTileCoordinates {
    double source_x_0;
    double source_y_0;
    int source_width;
    int source_height;
    double dem_x_0;
    double dem_y_0;
    int dem_width;
    int dem_height;
    double target_x_0;
    double target_y_0;
    int target_width;
    int target_height;
};

struct TcTile {
    alus::TcTileCoordinates tc_tile_coordinates;
    alus::cuda::KernelArray<double> target_tile_data_buffer;
    alus::cuda::KernelArray<double> source_tile_data_buffer;
    //alus::cuda::KernelArray<double> dem_tile_data_buffer;
    //alus::cuda::KernelArray<double> elevation_tile_data_buffer;
    alus::snapengine::resampling::TileData tile_data;
};
}