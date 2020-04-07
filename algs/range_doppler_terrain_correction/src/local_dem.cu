
#include <cstdio>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

#include "local_dem.cuh"

/**
 * This is a duplication of Dem::getLocalDemFor().
 *
 * @arg dem Elevation map
 * @arg targetElevations Elevation map for the product
 * @arg args Elevation and product raster size, geotransform consts etc.
 */
__global__ void fillElevation(double const* dem, double* targetElevations,
                              LocalDemKernelArgs const args) {
    int const threadX = threadIdx.x + blockIdx.x * blockDim.x;
    int const threadY = threadIdx.y + blockIdx.y * blockDim.y;

    if (threadX >= args.targetCols || threadY >= args.targetRows) {
        return;
    }

    // Dataset::getPixelCoordinatesFromIndex()
    double const targetLon =
        threadX * args.targetPixelSizeLon + args.targetOriginLon;
    double const targetLat =
        threadY * args.targetPixelSizeLat + args.targetOriginLat;

    // Dataset::getPixelIndexFromCoordinates()
    int const demX = (targetLon - args.demOriginLon) / args.demPixelSizeLon;
    int const demY = (targetLat - args.demOriginLat) / args.demPixelSizeLat;
    if (demX >= args.demCols || demY >= args.demRows || demX < 0 || demY < 0) {
        printf(
            "Index error in getElevation() kernel \"DEM index > size\" %d %d\n",
            demX, demY);
        return;
    }

    double const elevation = dem[args.demCols * demY + demX];
    targetElevations[args.targetCols * threadY + threadX] =
        (elevation > 0) * elevation;
}

void runElevationKernel(double const* dem, double* targetElevations,
                        LocalDemKernelArgs const args) {
    dim3 blockSize{32, 32};
    dim3 gridSize{args.targetCols / blockSize.x + 1,
                  args.targetRows / blockSize.y + 1};

    printf("Running kernel threads per block X:%d Y:%d blocks X:%d Y:%d\n",
           blockSize.x, blockSize.y, gridSize.x, gridSize.y);

    fillElevation<<<gridSize, blockSize>>>(dem, targetElevations, args);
}
