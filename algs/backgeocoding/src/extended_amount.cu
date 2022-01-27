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
#include "extended_amount_computation.h"

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>
#include <thrust/sequence.h>

#include "allocators.h"
#include "cuda_copies.h"
#include "cuda_ptr.h"
#include "extended_amount_computation.h"
#include "general_constants.h"
#include "pointer_holders.h"
#include "position_data.h"
#include "raster_properties.h"
#include "snap-dem/dem/dataio/earth_gravitational_model96_computation.h"

#include "../../../snap-engine/srtm3_elevation_calc.cuh"
#include "backgeocoding_utils.cuh"
#include "cuda_util.cuh"
#include "s1tbx-commons/sentinel1_utils.cuh"
#include "snap-dem/dem/dataio/earth_gravitational_model96.cuh"
#include "snap-engine-utilities/engine-utilities/eo/geo_utils.cuh"

namespace alus {
namespace backgeocoding {

namespace {
constexpr int index_step{20};

struct ExtendedAmountKernelArgs {
    s1tbx::DeviceSentinel1Utils* sentinel_utils;
    s1tbx::DeviceSubswathInfo* subswath_info;
    float* egm;
    Rectangle bounds;
    double* latitudes;
    double* longitudes;
    double* subswath_slant_range_times{nullptr};
    double* subswath_azimuth_times;
    snapengine::OrbitStateVectorComputation* orbit_state_vectors;
    size_t nr_of_orbit_vectors;
    PointerArray tiles;
    double dt;
    size_t col_count;
    size_t row_count;
};

__global__ void ComputeExtendedAmountKernel(ExtendedAmountKernelArgs args, AzimuthAndRangeBounds* result) {
    const size_t index_x = threadIdx.x + blockDim.x * blockIdx.x;
    const size_t index_y = threadIdx.y + blockDim.y * blockIdx.y;

    const size_t pixel_index_x{index_x * index_step};
    const size_t pixel_index_y{index_y * index_step};

    if (index_x >= args.col_count || index_y >= args.row_count) {
        return;
    }
    const size_t x = args.bounds.x + pixel_index_x;
    const size_t y = args.bounds.y + pixel_index_y;

    int burst_index = GetBurstIndex(y, args.subswath_info->num_of_bursts, args.subswath_info->lines_per_burst);

    double azimuth_time = GetAzimuthTime(y, burst_index, args.subswath_info);
    double slant_range_time = GetSlantRangeTime(x, args.subswath_info);

    s1tbx::Sentinel1Index sentinel_index =
        s1tbx::ComputeIndex(azimuth_time, slant_range_time, args.subswath_info, args.subswath_slant_range_times,
                            args.subswath_azimuth_times);

    double latitude =
        s1tbx::GetLatitude(sentinel_index, args.latitudes, args.subswath_info->num_of_geo_points_per_line);
    double longitude =
        s1tbx::GetLongitude(sentinel_index, args.longitudes, args.subswath_info->num_of_geo_points_per_line);
    double altitude = snapengine::srtm3elevationmodel::GetElevation(latitude, longitude, &args.tiles);
    // TODO: we may have to rewire this in the future, but no idea to where atm.
    //       (see algs/backgeocoding/src/backgeocoding.cc)
    if (altitude == snapengine::srtm3elevationmodel::NO_DATA_VALUE) {
        altitude = snapengine::earthgravitationalmodel96computation::GetEGM96(
            latitude, longitude, snapengine::earthgravitationalmodel96computation::MAX_LATS,
            snapengine::earthgravitationalmodel96computation::MAX_LONS, args.egm);
    }
    s1tbx::PositionData position_data{};
    snapengine::geoutils::Geo2xyzWgs84Impl(latitude, longitude, altitude, position_data.earth_point);
    if (backgeocoding::GetPosition(args.subswath_info, args.sentinel_utils, burst_index, &position_data,
                                   args.orbit_state_vectors, args.nr_of_orbit_vectors, args.dt, x, y)) {
        const int azimuth_extended_amount = static_cast<int>(position_data.azimuth_index - y);
        const int range_extended_amount = static_cast<int>(position_data.range_index - x);

        atomicMin(&result->azimuth_min, azimuth_extended_amount);
        atomicMax(&result->azimuth_max, azimuth_extended_amount);
        atomicMin(&result->range_min, range_extended_amount);
        atomicMax(&result->range_max, range_extended_amount);
    }
}

void PrepareArguments(ExtendedAmountKernelArgs* args, PointerArray tiles,
                      snapengine::OrbitStateVectorComputation* d_orbit_state_vectors, size_t nr_of_vectors,
                      double vectors_dt, const s1tbx::SubSwathInfo& subswath_info,
                      s1tbx::DeviceSentinel1Utils* d_sentinel_1_utils, s1tbx::DeviceSubswathInfo* d_subswath_info,
                      Rectangle& bounds, float* egm) {
    args->tiles.array = tiles.array;
    args->tiles.size = tiles.size;

    args->nr_of_orbit_vectors = nr_of_vectors;
    args->dt = vectors_dt;
    args->bounds = bounds;

    args->orbit_state_vectors = d_orbit_state_vectors;
    args->nr_of_orbit_vectors = nr_of_vectors;
    args->subswath_azimuth_times = subswath_info.devicePointersHolder.device_subswath_azimuth_times;
    args->subswath_slant_range_times = subswath_info.devicePointersHolder.device_subswath_slant_range_times;
    args->latitudes = subswath_info.devicePointersHolder.device_latidudes;
    args->longitudes = subswath_info.devicePointersHolder.device_longitudes;

    args->col_count = cuda::GetGridDim(index_step, bounds.width);
    args->row_count = cuda::GetGridDim(index_step, bounds.height);

    args->sentinel_utils = d_sentinel_1_utils;
    args->subswath_info = d_subswath_info;

    args->egm = egm;
}

}  // namespace

cudaError_t LaunchComputeExtendedAmount(Rectangle bounds, AzimuthAndRangeBounds& extended_amount,
                                        snapengine::OrbitStateVectorComputation* d_vectors, size_t nr_of_vectors,
                                        double vectors_dt, const s1tbx::SubSwathInfo& subswath_info,
                                        s1tbx::DeviceSentinel1Utils* d_sentinel_1_utils,
                                        s1tbx::DeviceSubswathInfo* d_subswath_info, const PointerArray& tiles,
                                        float* egm) {
    ExtendedAmountKernelArgs args{};

    const int idx_max = std::numeric_limits<int>::max();
    const int idx_min = std::numeric_limits<int>::lowest();

    AzimuthAndRangeBounds h_az_rg_bounds = {};
    h_az_rg_bounds.range_min = h_az_rg_bounds.azimuth_min = idx_max;
    h_az_rg_bounds.range_max = h_az_rg_bounds.azimuth_max = idx_min;

    cuda::CudaPtr<AzimuthAndRangeBounds> d_az_rg_bounds(1);
    cuda::CopyH2D(d_az_rg_bounds.Get(), &h_az_rg_bounds);

    PrepareArguments(&args, tiles, d_vectors, nr_of_vectors, vectors_dt, subswath_info, d_sentinel_1_utils,
                     d_subswath_info, bounds, egm);

    dim3 block_dim{20, 20};
    dim3 grid_dim(cuda::GetGridDim(block_dim.x, cuda::GetGridDim(index_step, bounds.width)),
                  cuda::GetGridDim(block_dim.y, cuda::GetGridDim(index_step, bounds.height)));
    ComputeExtendedAmountKernel<<<grid_dim, block_dim>>>(args, d_az_rg_bounds.Get());

    cuda::CopyD2H(&h_az_rg_bounds, d_az_rg_bounds.Get());
    cudaError error = cudaGetLastError();

    const int azimuth_min = h_az_rg_bounds.azimuth_min;
    const int azimuth_max = h_az_rg_bounds.azimuth_max;
    const int range_min = h_az_rg_bounds.range_min;
    const int range_max = h_az_rg_bounds.range_max;
    extended_amount.azimuth_min = (azimuth_min != idx_max && azimuth_min < 0) ? azimuth_min : 0;
    extended_amount.azimuth_max = (azimuth_max != idx_min && azimuth_max > 0) ? azimuth_max : 0;
    extended_amount.range_min = (range_min != idx_max && range_min < 0) ? range_min : 0;
    extended_amount.range_max = (range_max != idx_min && range_max > 0) ? range_max : 0;

    return error;
}
}  // namespace backgeocoding
}  // namespace alus
