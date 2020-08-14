#pragma once
#include "pointer_holders.h"

#include "resampling.cuh"

namespace alus {
namespace snapengine {
namespace bilinearinterpolation {

/**
        The 2 functions you see here are both from BilinearInterpolationResampling class in snap-engine.
        Resample functions requires a function pointer to a get samples function that can get the samples
        from the given raster. It seems that there are very different types of rasters in snap, so there
        are also very different type of getSamples functions. An example can be found in BaseElevationModel
        class in snap. However keep in mind that getSample and getSamples are 2 different functions.

        examples in our project can be found in algs/backgeocoding/src/bilinear.cu and
        util/snap-engine/include/srtm3_elevation_calc.cuh
*/
inline __device__ void ComputeIndex(
    const double x, const double y, const int width, const int height, snapengine::resampling::ResamplingIndex *index) {
    index->x = x;
    index->y = y;
    index->width = width;
    index->height = height;

    const int i0 = (int)x;
    const int j0 = (int)y;
    double di = x - (i0 + 0.5);
    double dj = y - (j0 + 0.5);

    index->i0 = i0;
    index->j0 = j0;

    int i_max = width - 1;
    int j_max = 0;

    if (di >= 0.0) {
        j_max = i0 + 1;
        index->i[0] = i0 < 0 ? 0.0 : (i0 > i_max ? i_max : i0);
        index->i[1] = j_max < 0 ? 0.0 : (j_max > i_max ? i_max : j_max);
        index->ki[0] = di;
    } else {
        j_max = i0 - 1;
        index->i[0] = j_max < 0 ? 0.0 : (j_max > i_max ? i_max : j_max);
        index->i[1] = i0 < 0 ? 0.0 : (i0 > i_max ? i_max : i0);
        index->ki[0] = di + 1.0;
    }

    j_max = height - 1;
    int j1 = 0;

    if (dj >= 0.0) {
        j1 = j0 + 1;
        index->j[0] = j0 < 0 ? 0.0 : (j0 > j_max ? j_max : j0);
        index->j[1] = j1 < 0 ? 0.0 : (j1 > j_max ? j_max : j1);
        index->kj[0] = dj;
    } else {
        j1 = j0 - 1;
        index->j[0] = j1 < 0 ? 0.0 : (j1 > j_max ? j_max : j1);
        index->j[1] = j0 < 0 ? 0.0 : (j0 > j_max ? j_max : j0);
        index->kj[0] = dj + 1.0;
    }
}

inline __device__ double Resample(
    PointerArray *tiles,
    snapengine::resampling::ResamplingIndex *index,
    int width,
    double no_value,
    int use_no_data,
    int GetSamplesFunction(PointerArray *, int *, int *, double *, int, int, double, int)) {
    int x[2] = {(int)index->i[0], (int)index->i[1]};
    int y[2] = {(int)index->j[0], (int)index->j[1]};
    double samples[2][2];
    samples[0][0] = 0.0;

    if (GetSamplesFunction(tiles, x, y, samples[0], width, 2, no_value, use_no_data)) {
        const double ki = index->ki[0];
        const double kj = index->kj[0];
        return samples[0][0] * (1.0 - ki) * (1.0 - kj) + samples[0][1] * ki * (1.0 - kj) +
               samples[1][0] * (1.0 - ki) * kj + samples[1][1] * ki * kj;
    } else {
        return samples[0][0];
    }
}
}  // namespace bilinearinterpolation
}  // namespace snapengine
}  // namespace alus
