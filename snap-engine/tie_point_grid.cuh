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

#include <cuda_runtime.h>

#include "math_utils.cuh"

#include "tie_point_grid.h"

namespace alus {
namespace snapengine {
namespace tiepointgrid {

inline __device__ __host__ double Interpolate(double wi, double wj, int i0, int j0,
                                              const tiepointgrid::TiePointGrid* grid) {
    const int w = grid->grid_width;
    const int j1 = j0 + 1;
    const int i1 = i0 + 1;

    return mathutils::Interpolate2D(wi, wj, grid->tie_points[i0 + j0 * w], grid->tie_points[i1 + j0 * w],
                                    grid->tie_points[i0 + j1 * w], grid->tie_points[i1 + j1 * w]);
}

inline __device__ __host__ double GetPixelDoubleImpl(double x, double y, const tiepointgrid::TiePointGrid* grid) {
    const double fi = (x - grid->offset_x) / grid->sub_sampling_x;
    const double fj = (y - grid->offset_y) / grid->sub_sampling_y;

    const int i = mathutils::FloorAndCrop(fi, 0, grid->grid_width - 2);
    const int j = mathutils::FloorAndCrop(fj, 0, grid->grid_height - 2);

    return Interpolate(fi - i, fj - j, i, j, grid);
}

}  // namespace tiepointgrid
}  // namespace snapengine
}  // namespace alus