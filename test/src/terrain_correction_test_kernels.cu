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
#include "cuda_util.cuh"
#include "cuda_util.h"
#include "get_position.cuh"
#include "get_position.h"

namespace alus {
namespace tests {

__global__ void GetPositionKernel(cuda::KernelArray<double> lat_args, cuda::KernelArray<double> lon_args,
                                  cuda::KernelArray<double> alt_args,
                                  cuda::KernelArray<s1tbx::PositionData> sat_positions,
                                  terraincorrection::GetPositionMetadata metadata, cuda::KernelArray<bool> results) {
    auto const thread_x = threadIdx.x + blockIdx.x * blockDim.x;

    if (thread_x >= lat_args.size) {
        return;
    }

    results.array[thread_x] =
        terraincorrection::GetPositionImpl(lat_args.array[thread_x], lon_args.array[thread_x], alt_args.array[thread_x],
                                           sat_positions.array[thread_x], metadata);
}

void LaunchGetPositionKernel(const std::vector<double>& lat_args, const std::vector<double>& lon_args,
                             const std::vector<double>& alt_args, std::vector<s1tbx::PositionData>& sat_positions,
                             terraincorrection::GetPositionMetadata metadata,
                             const std::vector<snapengine::PosVector>& sensor_position,
                             const std::vector<snapengine::PosVector>& sensor_velocity,
                             const std::vector<snapengine::OrbitStateVectorComputation>& orbit_state_vector,
                             std::vector<bool>& results) {
    dim3 block_size{32};
    dim3 grid_size{1};

    thrust::device_vector<double> d_lat_args = lat_args;
    cuda::KernelArray<double> k_lat_args{thrust::raw_pointer_cast(d_lat_args.data()), d_lat_args.size()};
    thrust::device_vector<double> d_lon_args = lon_args;
    cuda::KernelArray<double> k_lon_args{thrust::raw_pointer_cast(d_lon_args.data()), d_lon_args.size()};
    thrust::device_vector<double> d_alt_args = alt_args;
    cuda::KernelArray<double> k_alt_args{thrust::raw_pointer_cast(d_alt_args.data()), d_alt_args.size()};
    thrust::device_vector<s1tbx::PositionData> d_sat_positions(sat_positions.size());
    cuda::KernelArray<s1tbx::PositionData> k_sat_positions{thrust::raw_pointer_cast(d_sat_positions.data()),
                                                           d_sat_positions.size()};

    thrust::device_vector<double> d_osv_lookup = terraincorrection::CalculateOrbitStateVectorLUT(orbit_state_vector);

    thrust::device_vector<snapengine::PosVector> d_sensor_position = sensor_position;
    cuda::KernelArray<snapengine::PosVector> k_sensor_position{thrust::raw_pointer_cast(d_sensor_position.data()),
                                                               d_sensor_position.size()};
    thrust::device_vector<snapengine::PosVector> d_sensor_velocity = sensor_velocity;
    cuda::KernelArray<snapengine::PosVector> k_sensor_velocity{thrust::raw_pointer_cast(d_sensor_velocity.data()),
                                                               d_sensor_velocity.size()};
    thrust::device_vector<snapengine::OrbitStateVectorComputation> d_orbit_state_vector = orbit_state_vector;
    cuda::KernelArray<snapengine::OrbitStateVectorComputation> k_orbit_state_vector{
        thrust::raw_pointer_cast(d_orbit_state_vector.data()), d_orbit_state_vector.size()};

    cuda::KernelArray<double> k_osv_lookup = {d_osv_lookup.data().get(), d_osv_lookup.size()};
    auto k_metadata = metadata;
    k_metadata.sensor_position = k_sensor_position;
    k_metadata.sensor_velocity = k_sensor_velocity;
    k_metadata.orbit_state_vectors = k_orbit_state_vector;
    k_metadata.orbit_state_vector_lut = k_osv_lookup;

    thrust::device_vector<bool> d_results(results.size());
    cuda::KernelArray<bool> k_results{thrust::raw_pointer_cast(d_results.data()), d_results.size()};

    GetPositionKernel<<<grid_size, block_size>>>(k_lat_args, k_lon_args, k_alt_args, k_sat_positions, k_metadata,
                                                 k_results);
    CHECK_CUDA_ERR(cudaGetLastError());

    thrust::copy(d_sat_positions.begin(), d_sat_positions.end(), sat_positions.begin());
    thrust::copy(d_results.begin(), d_results.end(), results.begin());
}

}  // namespace tests
}  // namespace alus
