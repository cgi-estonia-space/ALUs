
#include <cstdio>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

#include "local_dem.cuh"

/**
 * This is a duplication of Dem::GetLocalDemFor().
 *
 * @arg dem Elevation map
 * @arg targetElevations Elevation map for the product
 * @arg args Elevation and product raster size, geotransform consts etc.
 */
__global__ void fillElevation(double const* dem, double* targetElevations,
                              LocalDemKernelArgs const args) {
    auto const threadX = threadIdx.x + blockIdx.x * blockDim.x;
    auto const threadY = threadIdx.y + blockIdx.y * blockDim.y;

    if (threadX >= args.target_cols || threadY >= args.target_rows) {
        return;
    }

    // Dataset::GetPixelCoordinatesFromIndex()
    double const targetLon =
        threadX * args.target_pixel_size_lon + args.target_origin_lon;
    double const targetLat =
        threadY * args.target_pixel_size_lat + args.target_origin_lat;

    // Dataset::GetPixelIndexFromCoordinates()
    auto const demX = static_cast<int>((targetLon - args.dem_origin_lon) / args.dem_pixel_size_lon);
    auto const demY = static_cast<int>((targetLat - args.dem_origin_lat) / args.dem_pixel_size_lat);
    if (demX >= args.dem_cols || demY >= args.dem_rows || demX < 0 || demY < 0) {
        printf(
            "Index error in getElevation() kernel \"DEM index > size\" %d %d\n",
            demX, demY);
        return;
    }

    double const elevation = dem[args.dem_cols * demY + demX];
    targetElevations[args.target_cols * threadY + threadX] =
        (elevation > 0) * elevation;
}

void RunElevationKernel(double const* dem, double* target_elevations,
                        LocalDemKernelArgs const args) {
    dim3 blockSize{32, 32};
    dim3 gridSize{args.target_cols / blockSize.x + 1,
                  args.target_rows / blockSize.y + 1};

    printf("Running kernel threads per block X:%d Y:%d blocks X:%d Y:%d\n",
           blockSize.x, blockSize.y, gridSize.x, gridSize.y);

    fillElevation<<<gridSize, blockSize>>>(dem, target_elevations, args);
}
