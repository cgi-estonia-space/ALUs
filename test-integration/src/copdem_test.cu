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

#include <cuda.h>

#include "copdem_cog_30m_calc.cuh"
#include "cuda_util.h"
#include "dem_property.h"
#include "pointer_holders.h"

__global__ void GetElevationWrapperKernel(double lat, double lon, alus::PointerArray p_array,
                                          const alus::dem::Property* dem_prop, double* result) {
    *result = alus::dem::CopDemCog30mGetElevation(lat, lon, &p_array, dem_prop);
}

__host__ double GetElevationWrapper(double lon, double lat, alus::PointerArray p_array,
                                    const alus::dem::Property* dem_prop) {
    double* dev_result{};
    CHECK_CUDA_ERR(cudaMalloc(&dev_result, sizeof(double)));
    GetElevationWrapperKernel<<<1, 1>>>(lat, lon, p_array, dem_prop, dev_result);
    CHECK_CUDA_ERR(cudaDeviceSynchronize());
    CHECK_CUDA_ERR(cudaGetLastError());
    double result{};
    CHECK_CUDA_ERR(cudaMemcpy(&result, dev_result, sizeof(double), cudaMemcpyDeviceToHost));

    return result;
}