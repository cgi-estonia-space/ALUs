
#include "cuda_util.cuh"
#include "cuda_util.hpp"
#include "get_position.cuh"
#include "get_position.h"

namespace alus {
namespace tests {

__global__ void GetPositionKernel(cudautil::KernelArray<double> lat_args,
                                  cudautil::KernelArray<double> lon_args,
                                  cudautil::KernelArray<double> alt_args,
                                  cudautil::KernelArray<s1tbx::PositionData> sat_positions,
                                  terraincorrection::GetPositionMetadata metadata,
                                  cudautil::KernelArray<bool> results) {
    auto const thread_x = threadIdx.x + blockIdx.x * blockDim.x;

    if (thread_x >= lat_args.size) {
        return;
    }

    results.array[thread_x] = terraincorrection::GetPositionImpl(lat_args.array[thread_x],
                                                                 lon_args.array[thread_x],
                                                                 alt_args.array[thread_x],
                                                                 sat_positions.array[thread_x],
                                                                 metadata);
}

void LaunchGetPositionKernel(const std::vector<double>& lat_args,
                             const std::vector<double>& lon_args,
                             const std::vector<double>& alt_args,
                             std::vector<s1tbx::PositionData>& sat_positions,
                             terraincorrection::GetPositionMetadata metadata,
                             const std::vector<snapengine::PosVector>& sensor_position,
                             const std::vector<snapengine::PosVector>& sensor_velocity,
                             const std::vector<snapengine::OrbitStateVector>& orbit_state_vector,
                             std::vector<bool>& results) {
    dim3 block_size{32};
    dim3 grid_size{1};

    thrust::device_vector<double> d_lat_args = lat_args;
    cudautil::KernelArray<double> k_lat_args{thrust::raw_pointer_cast(d_lat_args.data()), d_lat_args.size()};
    thrust::device_vector<double> d_lon_args = lon_args;
    cudautil::KernelArray<double> k_lon_args{thrust::raw_pointer_cast(d_lon_args.data()), d_lon_args.size()};
    thrust::device_vector<double> d_alt_args = alt_args;
    cudautil::KernelArray<double> k_alt_args{thrust::raw_pointer_cast(d_alt_args.data()), d_alt_args.size()};
    thrust::device_vector<s1tbx::PositionData> d_sat_positions(sat_positions.size());
    cudautil::KernelArray<s1tbx::PositionData> k_sat_positions{
        thrust::raw_pointer_cast(d_sat_positions.data()), d_sat_positions.size()};

    thrust::device_vector<snapengine::PosVector> d_sensor_position = sensor_position;
    cudautil::KernelArray<snapengine::PosVector> k_sensor_position{thrust::raw_pointer_cast(d_sensor_position.data()),
                                                                   d_sensor_position.size()};
    thrust::device_vector<snapengine::PosVector> d_sensor_velocity = sensor_velocity;
    cudautil::KernelArray<snapengine::PosVector> k_sensor_velocity{thrust::raw_pointer_cast(d_sensor_velocity.data()),
                                                                   d_sensor_velocity.size()};
    thrust::device_vector<snapengine::OrbitStateVector> d_orbit_state_vector = orbit_state_vector;
    cudautil::KernelArray<snapengine::OrbitStateVector> k_orbit_state_vector{
        thrust::raw_pointer_cast(d_orbit_state_vector.data()), d_orbit_state_vector.size()};
    auto k_metadata = metadata;
    k_metadata.sensor_position = k_sensor_position;
    k_metadata.sensor_velocity = k_sensor_velocity;
    k_metadata.orbit_state_vector = k_orbit_state_vector;

    thrust::device_vector<bool> d_results(results.size());
    cudautil::KernelArray<bool> k_results{thrust::raw_pointer_cast(d_results.data()), d_results.size()};

    GetPositionKernel<<<grid_size, block_size>>>(
        k_lat_args, k_lon_args, k_alt_args, k_sat_positions, k_metadata, k_results);
    CHECK_CUDA_ERR(cudaGetLastError());

    thrust::copy(d_sat_positions.begin(), d_sat_positions.end(), sat_positions.begin());
    thrust::copy(d_results.begin(), d_results.end(), results.begin());
}

}  // namespace tests
}  // namespace alus
