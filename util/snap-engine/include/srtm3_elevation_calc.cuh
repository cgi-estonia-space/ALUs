#pragma once

#include <cstdio>

#include "bilinear_interpolation.cuh"
#include "math_constants.h"  //not sure if required
#include "srtm3_elevation_model_constants.h"

namespace alus {
namespace snapengine {
namespace srtm3elevationmodel {

inline __device__ int getSamples(
    PointerArray *tiles, int *x, int *y, double *samples, int width, int height, double noValue, int useNoData) {
    // in this case, it will always be used.
    noValue = noValue;
    int allValid = 1;
    int i = 0, j = 0;
    int tileYIndex, tileXIndex, pixelY, pixelX;
    int xI, yI;
    double *srtm41_01Tile = (double *)tiles->array[0].pointer;
    int xSize = tiles->array[0].x;
    double *srtm42_01Tile = (double *)tiles->array[1].pointer;

    for (yI = 0; yI < height; yI++) {
        tileYIndex = (int)(y[yI] * NUM_PIXELS_PER_TILEinv);
        pixelY = y[yI] - tileYIndex * NUM_PIXELS_PER_TILE;

        j = 0;
        for (xI = 0; xI < width; xI++) {
            tileXIndex = (int)(x[xI] * NUM_PIXELS_PER_TILEinv);

            // final ElevationTile tile = elevationFiles[tileXIndex][tileYIndex].getTile();
            // make sure that the tile we want is actually listed
            if (tileXIndex > NUM_X_TILES || tileXIndex < 0 || tileYIndex > NUM_Y_TILES || tileYIndex < 0) {
                samples[i * width + j] = CUDART_NAN;
                allValid = 0;
                ++j;
                continue;
            }
            pixelX = x[xI] - tileXIndex * NUM_PIXELS_PER_TILE;

            // TODO: placeholder. Chanage once you know how dynamic tiling will work.
            switch (tileXIndex) {
                case 40:
                    samples[i * width + j] = srtm41_01Tile[pixelX + xSize * pixelY];
                    break;
                case 41:
                    samples[i * width + j] = srtm42_01Tile[pixelX + xSize * pixelY];
                    break;
                default:
                    printf("Slave pix pos where it should not be. %d \n", tileXIndex);
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

inline __device__ double getElevation(double geoPosLat, double geoPosLon, PointerArray *pArray) {
    double indexI[2];
    double indexJ[2];
    double indexKi[1];
    double indexKj[1];

    if (geoPosLon > 180) {
        geoPosLat -= 360;
    }

    double pixelY = (60.0 - geoPosLat) * DEGREE_RES_BY_NUM_PIXELS_PER_TILEinv;
    if (pixelY < 0 || isnan(pixelY)) {
        return NO_DATA_VALUE;
    }
    double pixelX = (geoPosLon + 180.0) * DEGREE_RES_BY_NUM_PIXELS_PER_TILEinv;
    double elevation = 0.0;

    // computing corner based index.
    snapengine::bilinearinterpolation::computeIndex(
        pixelX + 0.5, pixelY + 0.5, RASTER_WIDTH, RASTER_HEIGHT, indexI, indexJ, indexKi, indexKj);

    elevation = snapengine::bilinearinterpolation::resample(
        pArray, indexI, indexJ, indexKi, indexKj, CUDART_NAN, 1, getSamples);

    return isnan(elevation) ? NO_DATA_VALUE : elevation;
}

}  // namespace srtm3elevationmodel
}  // namespace snapengine
}  // namespace alus
