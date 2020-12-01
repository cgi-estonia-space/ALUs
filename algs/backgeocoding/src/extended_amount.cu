#include "extended_amount.h"

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>
#include <thrust/sequence.h>

#include "allocators.h"
#include "earth_gravitational_model96_computation.h"
#include "extended_amount.h"
#include "general_constants.h"
#include "pointer_holders.h"
#include "position_data.h"
#include "raster_properties.hpp"

#include "backgeocoding_utils.cuh"
#include "cuda_util.cuh"
#include "earth_gravitational_model96.cuh"
#include "geo_utils.cuh"
#include "sentinel1_utils.cuh"
#include "srtm3_elevation_calc.cuh"

namespace alus {
namespace backgeocoding {

constexpr int index_step {20};

struct ExtendedAmountKernelArgs {
    s1tbx::DeviceSentinel1Utils* sentinel_utils;
    s1tbx::DeviceSubswathInfo* subswath_info;
    float* egm;
    Rectangle bounds;
    double* latitudes;
    double* longitudes;
    double* subswath_slant_range_times{nullptr};
    double* subswath_azimuth_times;
    double* max_extended_azimuths;
    double* min_extended_azimuths;
    double* max_extended_ranges;
    double* min_extended_ranges;
    snapengine::OrbitStateVectorComputation* orbit_state_vectors;
    size_t nr_of_orbit_vectors;
    PointerArray tiles;
    double dt;
};

__global__ void ComputeExtendedAmountKernel(ExtendedAmountKernelArgs args) {
    const size_t index_x = threadIdx.x + blockDim.x * blockIdx.x;
    const size_t index_y = threadIdx.y + blockDim.y * blockIdx.y;

    const size_t pixel_index_x{index_x * index_step};
    const size_t pixel_index_y{index_y * index_step};

    const size_t n_cols = args.bounds.width / index_step;
    const size_t n_rows = args.bounds.height / index_step;

    if (index_x >= n_cols || index_y >= n_rows) {
        return;
    }
    const size_t x = args.bounds.x + pixel_index_x;
    const size_t y = args.bounds.y + pixel_index_y;
    const size_t thread_index = index_x + index_y * n_cols;

    int burst_index = GetBurstIndex(y, args.subswath_info->num_of_bursts, args.subswath_info->lines_per_burst);

    double azimuth_time = GetAzimuthTime(y, burst_index, args.subswath_info);
    double slant_range_time = GetSlantRangeTime(x, args.subswath_info);

    s1tbx::Sentinel1Index sentinel_index = s1tbx::ComputeIndex(azimuth_time,
                                                               slant_range_time,
                                                               args.subswath_info,
                                                               args.subswath_slant_range_times,
                                                               args.subswath_azimuth_times);

    double latitude =
        s1tbx::GetLatitude(sentinel_index, args.latitudes, args.subswath_info->num_of_geo_points_per_line);
    double longitude =
        s1tbx::GetLongitude(sentinel_index, args.longitudes, args.subswath_info->num_of_geo_points_per_line);
    double altitude = snapengine::srtm3elevationmodel::GetElevation(latitude, longitude, &args.tiles);
    // TODO: we may have to rewire this in the future, but no idea to where atm.
    //       (see algs/backgeocoding/src/backgeocoding.cc)
    if (altitude == snapengine::srtm3elevationmodel::NO_DATA_VALUE) {
        altitude = snapengine::earthgravitationalmodel96computation::GetEGM96(latitude,
                                                                   longitude,
                                                                   snapengine::earthgravitationalmodel96computation::MAX_LATS,
                                                                   snapengine::earthgravitationalmodel96computation::MAX_LONS,
                                                                   args.egm);
    }
    s1tbx::PositionData position_data{};
    snapengine::geoutils::Geo2xyzWgs84Impl(latitude, longitude, altitude, position_data.earth_point);
    if (backgeocoding::GetPosition(args.subswath_info,
                                   args.sentinel_utils,
                                   burst_index,
                                   &position_data,
                                   args.orbit_state_vectors,
                                   args.nr_of_orbit_vectors,
                                   args.dt,
                                   x,
                                   y)) {
        const double azimuth_extended_amount = position_data.azimuth_index - y;
        const double range_extended_amount = position_data.range_index - x;

        if (azimuth_extended_amount > args.max_extended_azimuths[thread_index]) {
            args.max_extended_azimuths[thread_index] = azimuth_extended_amount;
        }
        if (azimuth_extended_amount < args.min_extended_azimuths[thread_index]) {
            args.min_extended_azimuths[thread_index] = azimuth_extended_amount;
        }
        if (range_extended_amount > args.max_extended_ranges[thread_index]) {
            args.max_extended_ranges[thread_index] = range_extended_amount;
        }
        if (range_extended_amount < args.min_extended_ranges[thread_index]) {
            args.min_extended_ranges[thread_index] = range_extended_amount;
        }
    }
}

void PrepareArguments(ExtendedAmountKernelArgs* args,
                      PointerArray tiles,
                      const snapengine::OrbitStateVectorComputation* vectors,
                      size_t nr_of_vectors,
                      double vectors_dt,
                      const s1tbx::SubSwathInfo& subswath_info,
                      s1tbx::DeviceSentinel1Utils* d_sentinel_1_utils,
                      s1tbx::DeviceSubswathInfo* d_subswath_info,
                      thrust::device_vector<double>& max_extended_azimuths,
                      thrust::device_vector<double>& min_extended_azimuths,
                      thrust::device_vector<double>& max_extended_ranges,
                      thrust::device_vector<double>& min_extended_ranges,
                      Rectangle& bounds,
                      float* egm) {
    args->tiles.array = tiles.array;

    CHECK_CUDA_ERR(
        cudaMalloc(&args->orbit_state_vectors, sizeof(snapengine::OrbitStateVectorComputation) * nr_of_vectors));
    CHECK_CUDA_ERR(cudaMemcpy(args->orbit_state_vectors,
                              vectors,
                              sizeof(snapengine::OrbitStateVectorComputation) * nr_of_vectors,
                              cudaMemcpyHostToDevice));
    args->nr_of_orbit_vectors = nr_of_vectors;
    args->dt = vectors_dt;
    args->bounds = bounds;

    int subswath_geo_grid_size = subswath_info.num_of_geo_lines_ * subswath_info.num_of_geo_points_per_line_;

    CHECK_CUDA_ERR(cudaMalloc(&args->subswath_azimuth_times, sizeof(double) * subswath_geo_grid_size));
    CHECK_CUDA_ERR(cudaMalloc(&args->subswath_slant_range_times, sizeof(double) * subswath_geo_grid_size));
    CHECK_CUDA_ERR(cudaMalloc(&args->longitudes, sizeof(double) * subswath_geo_grid_size));
    CHECK_CUDA_ERR(cudaMalloc(&args->latitudes, sizeof(double) * subswath_geo_grid_size));

    CHECK_CUDA_ERR(cudaMemcpy(args->subswath_azimuth_times,
                              subswath_info.azimuth_time_[0],
                              subswath_geo_grid_size * sizeof(double),
                              cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(args->subswath_slant_range_times,
                              subswath_info.slant_range_time_[0],
                              subswath_geo_grid_size * sizeof(double),
                              cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(args->longitudes,
                              subswath_info.longitude_[0],
                              subswath_geo_grid_size * sizeof(double),
                              cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(
        args->latitudes, subswath_info.latitude_[0], subswath_geo_grid_size * sizeof(double), cudaMemcpyHostToDevice));

    args->max_extended_azimuths = thrust::raw_pointer_cast(max_extended_azimuths.data());
    args->min_extended_azimuths = thrust::raw_pointer_cast(min_extended_azimuths.data());
    args->max_extended_ranges = thrust::raw_pointer_cast(max_extended_ranges.data());
    args->min_extended_ranges = thrust::raw_pointer_cast(min_extended_ranges.data());

    args->sentinel_utils = d_sentinel_1_utils;
    args->subswath_info = d_subswath_info;

    args->egm = egm;
}

cudaError_t LaunchComputeExtendedAmount(Rectangle bounds,
                                        AzimuthAndRangeBounds& extended_amount,
                                        const snapengine::OrbitStateVectorComputation* vectors,
                                        size_t nr_of_vectors,
                                        double vectors_dt,
                                        const s1tbx::SubSwathInfo& subswath_info,
                                        s1tbx::DeviceSentinel1Utils* d_sentinel_1_utils,
                                        s1tbx::DeviceSubswathInfo* d_subswath_info,
                                        const PointerArray& tiles,
                                        float* egm) {
    ExtendedAmountKernelArgs args{};

    const double double_max = std::numeric_limits<double>::max();
    const double double_min = std::numeric_limits<double>::lowest();

    // Initialises thrust vectors for max and min extended amounts
    // Vectors are used in order to avoid using shared memory and atomic operations inside kernel
    int total_thread_count = (bounds.width / index_step) * (bounds.height / index_step);
    thrust::device_vector<double> max_extended_azimuths(total_thread_count, double_min);
    thrust::device_vector<double> min_extended_azimuths(total_thread_count, double_max);
    thrust::device_vector<double> max_extended_ranges(total_thread_count, double_min);
    thrust::device_vector<double> min_extended_ranges(total_thread_count, double_max);

    PrepareArguments(&args,
                     tiles,
                     vectors,
                     nr_of_vectors,
                     vectors_dt,
                     subswath_info,
                     d_sentinel_1_utils,
                     d_subswath_info,
                     max_extended_azimuths,
                     min_extended_azimuths,
                     max_extended_ranges,
                     min_extended_ranges,
                     bounds,
                     egm);

    dim3 block_dim{20, 20};
    dim3 grid_dim(cuda::GetGridDim(block_dim.x, bounds.width / index_step), cuda::GetGridDim(block_dim.y, bounds.height / index_step));
    ComputeExtendedAmountKernel<<<grid_dim, block_dim>>>(args);

    cudaDeviceSynchronize();
    cudaError error = cudaGetLastError();

    double const azimuth_max =
        *thrust::max_element(thrust::device, max_extended_azimuths.begin(), max_extended_azimuths.end());
    double const azimuth_min =
        *thrust::min_element(thrust::device, min_extended_azimuths.begin(), min_extended_azimuths.end());
    double const range_max =
        *thrust::max_element(thrust::device, max_extended_ranges.begin(), max_extended_ranges.end());
    double const range_min =
        *thrust::min_element(thrust::device, min_extended_ranges.begin(), min_extended_ranges.end());

    extended_amount.azimuth_min = (azimuth_min != double_max && azimuth_min < 0.0) ? azimuth_min : 0.0;
    extended_amount.azimuth_max = (azimuth_max != double_min && azimuth_max > 0.0) ? azimuth_max : 0.0;
    extended_amount.range_min = (range_min != double_max && range_min < 0.0) ? range_min : 0.0;
    extended_amount.range_max = (range_max != double_min && range_max > 0.0) ? range_max : 0.0;

    CHECK_CUDA_ERR(cudaFree(args.subswath_slant_range_times));
    CHECK_CUDA_ERR(cudaFree(args.subswath_azimuth_times));
    CHECK_CUDA_ERR(cudaFree(args.longitudes));
    CHECK_CUDA_ERR(cudaFree(args.latitudes));

    return error;
}
}  // namespace backgeocoding
}  // namespace alus