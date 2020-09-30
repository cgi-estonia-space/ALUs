
#include <cstdio>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <math_constants.h>

#include "bilinear_interpolation.cuh"
#include "cuda_util.hpp"
#include "local_dem.cuh"
#include "pointer_holders.h"
#include "raster_utils.cuh"
#include "resampling.h"

/**
 * This is a duplication of Dem::GetLocalDemFor().
 *
 * @arg dem Elevation map
 * @arg targetElevations Elevation map for the product
 * @arg args Elevation and product raster size, geotransform consts etc.
 * @todo Might be removed and replaced by SRTM3::GetElevation()
 */
__global__ void FillElevation(double const *dem, double *targetElevations, LocalDemKernelArgs const args) {
    auto const thread_x = threadIdx.x + blockIdx.x * blockDim.x;
    auto const thread_y = threadIdx.y + blockIdx.y * blockDim.y;

    if (thread_x >= args.target_width || thread_y >= args.target_height) {  // TODO: remove conflict here
        return;
    }

    // Dataset::getPixelCoordinatesFromIndex()
    double const TARGET_LON = (thread_x + args.target_x_0) * args.target_geo_transform.pixelSizeLon +
                              args.target_geo_transform.originLon + args.target_geo_transform.pixelSizeLon / 2;
    double const TARGET_LAT = (thread_y + args.target_y_0) * args.target_geo_transform.pixelSizeLat +
                              args.target_geo_transform.originLat + args.target_geo_transform.pixelSizeLat / 2;

    // Dataset::getPixelIndexFromCoordinates()
    auto dem_x = (TARGET_LON - args.dem_geo_transform.pixelSizeLon / 2 - args.dem_geo_transform.originLon) /
                 args.dem_geo_transform.pixelSizeLon;
    auto dem_y = (TARGET_LAT - args.dem_geo_transform.pixelSizeLat / 2 - args.dem_geo_transform.originLat) /
                 args.dem_geo_transform.pixelSizeLat;

    auto tile_x = dem_x - args.dem_x_0;
    auto tile_y = dem_y - args.dem_y_0;
    if (std::floor(tile_x) > args.dem_tile_width || std::floor(tile_y) > args.dem_tile_height) {
        printf(
            "Error! Dem pixel indices are out of bounds!\n"
            "Bounds: %.10f, %.10f. Indices: %.10f. %.10f\n",
            args.dem_tile_width + args.dem_x_0,
            args.dem_tile_height + args.dem_y_0,
            dem_x,
            dem_y);
        // TODO: throw exception
        return;
    }

    double const elevation = dem[args.dem_tile_width * (int)tile_y + (int)tile_x];
    targetElevations[args.target_width * thread_y + thread_x] = (elevation > 0) * elevation;
}

void RunElevationKernel(double const *dem, double *targetElevations, LocalDemKernelArgs const args) {
    dim3 block_size{32, 32};
    dim3 grid_size{(args.target_width) / block_size.x + 1, (args.target_height) / block_size.y + 1};

    FillElevation<<<grid_size, block_size>>>(dem, targetElevations, args);

    CHECK_CUDA_ERR(cudaGetLastError());
    CHECK_CUDA_ERR(cudaDeviceSynchronize());
}
