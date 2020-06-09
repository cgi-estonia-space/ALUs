#pragma once

#include <cuda_runtime.h>
#include "math_utils.cuh"
#include "tie_point_grid.h"

namespace alus {
namespace snapengine {
namespace tiepointgrid {

inline __device__ __host__ double Interpolate(
    double wi, double wj, int i0, int j0, const tiepointgrid::TiePointGrid *grid) {
    const int w = grid->grid_width;
    const int j1 = j0 + 1;
    const int i1 = i0 + 1;
    const cudautil::KernelArray<float> TIE_POINTS = grid->tie_points;

    return mathutils::Interpolate2D(wi,
                                    wj,
                                    TIE_POINTS.array[i0 + j0 * w],
                                    TIE_POINTS.array[i1 + j0 * w],
                                    TIE_POINTS.array[i0 + j1 * w],
                                    TIE_POINTS.array[i1 + j1 * w]);
}

inline __device__ __host__ double GetPixelDoubleImpl(double x, double y, const tiepointgrid::TiePointGrid *grid) {
    const double fi = (x - grid->offset_x) / grid->sub_sampling_x;
    const double fj = (y - grid->offset_y) / grid->sub_sampling_y;

    const int i = mathutils::FloorAndCrop(fi, 0, grid->grid_width - 2);
    const int j = mathutils::FloorAndCrop(fj, 0, grid->grid_height - 2);

    return Interpolate(fi - i, fj - j, i, j, grid);
}

}  // namespace tiepointgrid
}  // namespace snapengine
}  // namespace alus