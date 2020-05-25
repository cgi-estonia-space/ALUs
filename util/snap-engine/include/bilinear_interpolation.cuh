#pragma once
#include "pointer_holders.h"

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
inline __device__ void computeIndex(const double x,
                                    const double y,
                                    const int width,
                                    const int height,
                                    double *indexI,
                                    double *indexJ,
                                    double *indexKi,
                                    double *indexKj) {
    const int i0 = (int)x;
    const int j0 = (int)y;
    double di = x - (i0 + 0.5);
    double dj = y - (j0 + 0.5);

    int iMax = width - 1;
    int jMax = 0;

    if (di >= 0.0) {
        jMax = i0 + 1;
        indexI[0] = i0 < 0 ? 0.0 : (i0 > iMax ? iMax : i0);
        indexI[1] = jMax < 0 ? 0.0 : (jMax > iMax ? iMax : jMax);
        indexKi[0] = di;
    } else {
        jMax = i0 - 1;
        indexI[0] = jMax < 0 ? 0.0 : (jMax > iMax ? iMax : jMax);
        indexI[1] = i0 < 0 ? 0.0 : (i0 > iMax ? iMax : i0);
        indexKi[0] = di + 1.0;
    }

    jMax = height - 1;
    int j1 = 0;

    if (dj >= 0.0) {
        j1 = j0 + 1;
        indexJ[0] = j0 < 0 ? 0.0 : (j0 > jMax ? jMax : j0);
        indexJ[1] = j1 < 0 ? 0.0 : (j1 > jMax ? jMax : j1);
        indexKj[0] = dj;
    } else {
        j1 = j0 - 1;
        indexJ[0] = j1 < 0 ? 0.0 : (j1 > jMax ? jMax : j1);
        indexJ[1] = j0 < 0 ? 0.0 : (j0 > jMax ? jMax : j0);
        indexKj[0] = dj + 1.0;
    }
}

inline __device__ double resample(
        PointerArray *tiles,
        double *indexI,
        double *indexJ,
        double *indexKi,
        double *indexKj,
        double noValue,
        int useNoData,
        int getSamplesFunction(PointerArray *, int *, int *, double *, int, int, double, int)) {


    int x[2] = {(int)indexI[0], (int)indexI[1]};
    int y[2] = {(int)indexJ[0], (int)indexJ[1]};
    double samples[2][2];
    samples[0][0] = 0.0;
    
    if (getSamplesFunction(tiles, x, y, samples[0], 2, 2, noValue, useNoData)) {
        const double ki = indexKi[0];
        const double kj = indexKj[0];
        return samples[0][0] * (1.0 - ki) * (1.0 - kj) + samples[0][1] * ki * (1.0 - kj) +
               samples[1][0] * (1.0 - ki) * kj + samples[1][1] * ki * kj;
    } else {
        return samples[0][0];
    }
}

}  // namespace bilinearinterpolation
}  // namespace snapengine
}  // namespace alus
