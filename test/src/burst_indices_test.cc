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
#include <driver_types.h>

#include "gmock/gmock.h"

#include "burst_indices_computation.h"
#include "cuda_util.h"
#include "orbit_state_vector_computation.h"
#include "position_data.h"

// Forward declaration
namespace alus::tests {
cudaError_t LaunchComputeBurstIndicesKernel(double line_time_interval, double wavelength, int num_of_bursts,
                                            const double* burst_first_line_times, const double* burst_last_line_times,
                                            snapengine::PosVector earth_point,
                                            snapengine::OrbitStateVectorComputation* orbit, size_t num_orbit_vec,
                                            double dt, backgeocoding::BurstIndices* indices, dim3 grid_dim,
                                            dim3 block_dim);
}

namespace {

using alus::backgeocoding::BurstIndices;
using alus::snapengine::OrbitStateVectorComputation;
using alus::snapengine::PosVector;
using alus::tests::LaunchComputeBurstIndicesKernel;

class BurstIndicesTest : public ::testing::Test {
protected:
    const double line_time_interval_{2.3791160879629606E-8};
    const double wavelength_{0.05546576};
    const int num_of_bursts_{9};
    const std::vector<double> burst_first_line_times_{6.4887741417916e8,  6.488774169418269e8, 6.48877419696273e8,
                                                      6.48877422452774e8, 6.48877425209275e8,  6.48877427967831e8,
                                                      6.48877430726388e8, 6.48877433484944e8,  6.48877436243501e8};
    const std::vector<double> burst_last_line_times_{6.4887741726455e8,  6.488774200272169e8, 6.48877422781663e8,
                                                     6.48877425538164e8, 6.48877428294665e8,  6.48877431053221e8,
                                                     6.48877433811778e8, 6.48877436570334e8,  6.488774393288909e8};
    const std::vector<OrbitStateVectorComputation> orbit_{
        {7510.154510824572, 7510.154510824572, 3673383.6086, 4325991.863895, 4569.33705, 1647.673002, -5836.793471},
        {7510.154626565312, 4262854.414968, 3689620.922176, 4267380.762376, 4524.172986, 1599.770229, -5885.316687},
        {7510.154742306042, 4307867.913344, 3705378.632928, 4208287.742233, 4478.430706, 1551.753484, -5933.176272},
        {7510.154858046794, 4352421.118458, 3720655.632362, 4148719.467207, 4432.115192, 1503.629011, -5980.366771},
        {7510.154973787535, 4396508.32323, 3735450.874485, 4088682.655165, 4385.231501, 1455.403061, -6026.882802},
        {7510.155089528275, 4440123.871589, 3749763.375842, 4028184.077314, 4337.784763, 1407.081888, -6072.719067},
        {7510.155205269005, 4483262.159107, 3763592.215514, 3967230.557468, 4289.780179, 1358.671748, -6117.870341},
        {7510.155321009757, 4525917.633577, 3776936.535062, 3905828.971232, 4241.223023, 1310.178902, -6162.331481},
        {7510.155436750498, 4568084.795876, 3789795.538651, 3843986.245284, 4192.118638, 1261.60961, -6206.097419},
        {7510.155552491238, 4609758.200672, 3802168.493112, 3781709.356694, 4142.47244, 1212.970136, -6249.163164},
        {7510.155668231968, 4650932.457157, 3814054.727948, 3719005.332104, 4092.289915, 1164.266741, -6291.523806},
        {7510.15578397272, 4691602.229762, 3825453.635309, 3655881.246872, 4041.576618, 1115.50569, -6333.174511},
        {7510.155899713461, 4731762.238915, 3836364.669989, 3592344.224208, 3990.338176, 1066.693246, -6374.110525},
        {7510.156015454201, 4771407.261677, 3846787.349405, 3528401.434449, 3938.580287, 1017.83567, -6414.327173},
        {7510.156131194931, 4810532.132367, 3856721.253536, 3464060.094252, 3886.308716, 968.939222, -6453.819861},
        {7510.156246935683, 4849131.743282, 3866166.024925, 3399327.465796, 3833.529299, 920.010158, -6492.584077}};
    const size_t num_orb_vec_ = orbit_.size();
    const double dt_ = 1.1574074075421474e-4;

    const double* d_burst_first_line_times_;
    const double* d_burst_last_line_times_;
    const OrbitStateVectorComputation* d_orbit_;

public:
    BurstIndicesTest() {
        CHECK_CUDA_ERR(cudaMalloc((void**)&d_burst_first_line_times_, sizeof(double) * burst_first_line_times_.size()));
        CHECK_CUDA_ERR(cudaMalloc((void**)&d_burst_last_line_times_, sizeof(double) * burst_last_line_times_.size()));
        CHECK_CUDA_ERR(cudaMalloc((void**)&d_orbit_, sizeof(OrbitStateVectorComputation) * num_orb_vec_));

        CHECK_CUDA_ERR(cudaMemcpy((void*)d_burst_first_line_times_, burst_first_line_times_.data(),
                                  sizeof(double) * burst_first_line_times_.size(), cudaMemcpyHostToDevice));
        CHECK_CUDA_ERR(cudaMemcpy((void*)d_burst_last_line_times_, burst_last_line_times_.data(),
                                  sizeof(double) * burst_last_line_times_.size(), cudaMemcpyHostToDevice));
        CHECK_CUDA_ERR(cudaMemcpy((void*)d_orbit_, orbit_.data(), sizeof(OrbitStateVectorComputation) * orbit_.size(),
                                  cudaMemcpyHostToDevice));
    }

    ~BurstIndicesTest() override {
        cudaFree(reinterpret_cast<void*>(const_cast<double*>(d_burst_last_line_times_)));
        cudaFree(reinterpret_cast<void*>(const_cast<double*>(d_burst_first_line_times_)));
        cudaFree(reinterpret_cast<void*>(const_cast<OrbitStateVectorComputation*>(d_orbit_)));
    }
};
TEST_F(BurstIndicesTest, ComputeBurstIndices) {
    BurstIndices* d_computed_indices;
    CHECK_CUDA_ERR(cudaMalloc((void**)&d_computed_indices, sizeof(BurstIndices)));

    PosVector earth_point{4266449.299985306, 3092570.538750303, 3582073.5968776057};  // NOLINT
    CHECK_CUDA_ERR(LaunchComputeBurstIndicesKernel(line_time_interval_, wavelength_, num_of_bursts_,
                                                   d_burst_first_line_times_, d_burst_last_line_times_, earth_point,
                                                   const_cast<OrbitStateVectorComputation*>(d_orbit_), num_orb_vec_,
                                                   dt_, d_computed_indices, {1}, {1}));
    BurstIndices computed_indices[1];
    CHECK_CUDA_ERR(cudaMemcpy(&computed_indices, d_computed_indices, sizeof(BurstIndices), cudaMemcpyDeviceToHost));
    EXPECT_THAT(computed_indices[0].first_burst_index, ::testing::Eq(1));
    EXPECT_THAT(computed_indices[0].second_burst_index, ::testing::Eq(-1));
    EXPECT_THAT(computed_indices[0].in_upper_part_of_first_burst, ::testing::IsTrue());
    EXPECT_THAT(computed_indices[0].in_upper_part_of_second_burst, ::testing::IsFalse());
    EXPECT_THAT(computed_indices[0].valid, ::testing::IsTrue());
    CHECK_CUDA_ERR(cudaFree(d_computed_indices));
}

TEST_F(BurstIndicesTest, ComputeBurstIndicesInvalid) {
    BurstIndices* d_computed_indices;
    CHECK_CUDA_ERR(cudaMalloc((void**)&d_computed_indices, sizeof(BurstIndices)));

    PosVector earth_point{4244349.646061476, 3087456.945708205, 3612029.2389756213};  // NOLINT
    CHECK_CUDA_ERR(LaunchComputeBurstIndicesKernel(line_time_interval_, wavelength_, num_of_bursts_,
                                                   d_burst_first_line_times_, d_burst_last_line_times_, earth_point,
                                                   const_cast<OrbitStateVectorComputation*>(d_orbit_), num_orb_vec_,
                                                   dt_, d_computed_indices, {1}, {1}));
    BurstIndices computed_indices[1];
    CHECK_CUDA_ERR(cudaMemcpy(&computed_indices, d_computed_indices, sizeof(BurstIndices), cudaMemcpyDeviceToHost));
    EXPECT_THAT(computed_indices[0].first_burst_index, ::testing::Eq(-1));
    EXPECT_THAT(computed_indices[0].second_burst_index, ::testing::Eq(-1));
    EXPECT_THAT(computed_indices[0].in_upper_part_of_second_burst, ::testing::IsFalse());
    EXPECT_THAT(computed_indices[0].in_upper_part_of_first_burst, ::testing::IsFalse());
    EXPECT_THAT(computed_indices[0].valid, ::testing::IsFalse());
    CHECK_CUDA_ERR(cudaFree(d_computed_indices));
}
}  // namespace