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
#include "gmock/gmock.h"

#include <vector>

#include <cuda_runtime.h>
#include <driver_types.h>

#include "cuda_util.h"
#include "shapes.h"
#include "test_constants.h"
#include "test_expected_values.h"
#include "thermal_noise_kernel.h"
#include "thermal_noise_utils.h"

namespace {
namespace test = alus::tnr::test;
namespace tnr = alus::tnr;

class ThermalNoiseKernelTest : public ::testing::Test {
protected:
    cudaStream_t stream_;

    ThermalNoiseKernelTest() { CHECK_CUDA_ERR(cudaStreamCreate(&stream_)); }
    ~ThermalNoiseKernelTest() override { cudaStreamDestroy(stream_); }
};

TEST_F(ThermalNoiseKernelTest, interpolateNoiseAzimuthVector) {
    const int first_azimuth_line{0};
    const int last_azimuth_line{355};
    std::vector<double> interpolated_noise_azimuth_vector(356);

    const auto d_noise_azimuth_vector = test::expectedvalues::IW1_VV_NOISE_AZIMUTH_VECTOR_LIST.at(0).ToDeviceVector();

    const auto starting_line_index =
        tnr::GetLineIndex(
        first_azimuth_line, test::expectedvalues::IW1_VV_NOISE_AZIMUTH_VECTOR_LIST.at(0).lines);

    const auto d_interpolated = tnr::LaunchInterpolateNoiseAzimuthVectorKernel(
        d_noise_azimuth_vector, first_azimuth_line, last_azimuth_line, starting_line_index, stream_);

    CHECK_CUDA_ERR(cudaMemcpy(interpolated_noise_azimuth_vector.data(), d_interpolated.array,
                              sizeof(double) * d_interpolated.size, cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < d_interpolated.size; i++) {
        ASSERT_THAT(interpolated_noise_azimuth_vector.at(i),
                    ::testing::DoubleEq(test::expectedvalues::INTERPOLATED_NOISE_AZIMUTH_VECTOR.at(i)));
    }

    CHECK_CUDA_ERR(cudaFree(d_interpolated.array));
    CHECK_CUDA_ERR(cudaFree(d_noise_azimuth_vector.lines.array));
    CHECK_CUDA_ERR(cudaFree(d_noise_azimuth_vector.noise_azimuth_lut.array));
}

TEST_F(ThermalNoiseKernelTest, getSampleIndexTest) {
    const std::vector<alus::s1tbx::NoiseVector> noise_vectors{
        {6801.658508998079,
         0,
         {0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000},
         {}},
        {},
        {6801.6585409496065, 0, {1000, 2000, 3000, 4000, 5000, 6000, 7000}, {}},
        {}};
    const std::vector<alus::Rectangle> tiles{{700, 0, 500, 500}, {1400, 1000, 700, 480}, {2874, 15982, 99999, 1}};
    const std::vector<int> burst_indices{0, 2};
    const std::vector<std::vector<size_t>> expected_results{{1, 0}, {2, 0}, {5, 1}};

    std::vector<alus::s1tbx::DeviceNoiseVector> h_burst_to_range_map;
    h_burst_to_range_map.reserve(noise_vectors.size());

    for (const auto& vector : noise_vectors) {
        h_burst_to_range_map.emplace_back(vector.ToDeviceVector());
    }

    alus::cuda::KernelArray<alus::s1tbx::DeviceNoiseVector> d_burst_to_range_map{nullptr, h_burst_to_range_map.size()};
    CHECK_CUDA_ERR(cudaMalloc(&d_burst_to_range_map.array, d_burst_to_range_map.ByteSize()));
    CHECK_CUDA_ERR(cudaMemcpy(d_burst_to_range_map.array, h_burst_to_range_map.data(), d_burst_to_range_map.ByteSize(),
                              cudaMemcpyHostToDevice));

    alus::cuda::KernelArray<int> d_burst_indices{nullptr, burst_indices.size()};
    CHECK_CUDA_ERR(cudaMalloc(&d_burst_indices.array, d_burst_indices.ByteSize()));
    CHECK_CUDA_ERR(
        cudaMemcpy(d_burst_indices.array, burst_indices.data(), d_burst_indices.ByteSize(), cudaMemcpyHostToDevice));

    for (size_t i = 0; i < tiles.size(); i++) {
        const auto d_result =
            alus::tnr::LaunchGetSampleIndexKernel(tiles.at(i), d_burst_to_range_map, d_burst_indices, stream_);

        std::vector<size_t> h_result(d_result.size);
        CHECK_CUDA_ERR(cudaMemcpy(h_result.data(), d_result.array, d_result.ByteSize(), cudaMemcpyDeviceToHost));

        const auto& expected_values = expected_results.at(i);
        for (size_t j = 0; j < expected_values.size(); j++) {
            ASSERT_THAT(h_result.at(j), ::testing::Eq(expected_values.at(j)));
        }

        CHECK_CUDA_ERR(cudaFree(d_result.array));
    }

    for (auto& vector : h_burst_to_range_map) {
        CHECK_CUDA_ERR(cudaFree(vector.pixels.array));
        CHECK_CUDA_ERR(cudaFree(vector.noise_lut.array));
    }
    CHECK_CUDA_ERR(cudaFree(d_burst_to_range_map.array));
    CHECK_CUDA_ERR(cudaFree(d_burst_indices.array));
}

TEST_F(ThermalNoiseKernelTest, interpolateRangeVectorTest) {
    const std::vector<alus::Rectangle> input_tiles{{3852, 0, 428, 356}, {0, 1424, 428, 356}};
    const std::vector<int> burst_indices{0, 1};       // One index per tile.
    const std::vector<size_t> sample_indices{96, 0};  // One index per tile.

    ASSERT_THAT(input_tiles, ::testing::SizeIs(test::expectedvalues::IW1_VV_INTERPOLATED_RANGE_VECTORS.size()));
    ASSERT_THAT(burst_indices, ::testing::SizeIs(test::expectedvalues::IW1_VV_INTERPOLATED_RANGE_VECTORS.size()));
    ASSERT_THAT(sample_indices, ::testing::SizeIs(test::expectedvalues::IW1_VV_INTERPOLATED_RANGE_VECTORS.size()));

    std::vector<alus::s1tbx::DeviceNoiseVector> h_burst_to_range_index_map;
    h_burst_to_range_index_map.reserve(test::constants::KERNEL_TEST_NOISE_VECTORS.size());
    for (const auto& vector : test::constants::KERNEL_TEST_NOISE_VECTORS) {
        h_burst_to_range_index_map.emplace_back(vector.ToDeviceVector());
    }

    alus::tnr::device::BurstIndexToRangeVectorMap d_burst_index_to_range_vector_map{
        nullptr, h_burst_to_range_index_map.size()};
    CHECK_CUDA_ERR(cudaMalloc(&d_burst_index_to_range_vector_map.array, d_burst_index_to_range_vector_map.ByteSize()));
    CHECK_CUDA_ERR(cudaMemcpy(d_burst_index_to_range_vector_map.array, h_burst_to_range_index_map.data(),
                              d_burst_index_to_range_vector_map.ByteSize(), cudaMemcpyHostToDevice));

    for (size_t i = 0; i < input_tiles.size(); ++i) {
        alus::cuda::KernelArray<int> d_burst_indices{nullptr, 1};
        CHECK_CUDA_ERR(cudaMalloc(&d_burst_indices.array, d_burst_indices.ByteSize()));
        CHECK_CUDA_ERR(cudaMemcpy(d_burst_indices.array, burst_indices.data() + i, d_burst_indices.ByteSize(),
                                  cudaMemcpyHostToDevice));

        alus::cuda::KernelArray<size_t> d_sample_indices{nullptr, 1};
        CHECK_CUDA_ERR(cudaMalloc(&d_sample_indices.array, d_sample_indices.ByteSize()));
        CHECK_CUDA_ERR(cudaMemcpy(d_sample_indices.array, sample_indices.data() + i, d_sample_indices.ByteSize(),
                                  cudaMemcpyHostToDevice));

        auto d_map = alus::tnr::LaunchInterpolateNoiseRangeVectorsKernel(
            input_tiles.at(i), d_burst_indices, d_sample_indices, d_burst_index_to_range_vector_map, stream_);

        CHECK_CUDA_ERR(cudaDeviceSynchronize());
        CHECK_CUDA_ERR(cudaGetLastError());

        // Copies data back to CPU
        std::vector<alus::cuda::KernelArray<double>> intermediary_result(d_map.size);
        CHECK_CUDA_ERR(cudaMemcpy(intermediary_result.data(), d_map.array, d_map.ByteSize(), cudaMemcpyDeviceToHost));
        std::vector<std::vector<double>> result(d_map.size);
        for (size_t j = 0; j < result.size(); j++) {
            result.at(j).resize(intermediary_result.at(j).size);
            CHECK_CUDA_ERR(cudaMemcpy(result.at(j).data(), intermediary_result.at(j).array,
                                      intermediary_result.at(j).ByteSize(), cudaMemcpyDeviceToHost));
        }

        // Perform assertions
        ASSERT_THAT(result, ::testing::SizeIs(h_burst_to_range_index_map.size()));
        const auto& expected_vector = test::expectedvalues::IW1_VV_INTERPOLATED_RANGE_VECTORS.at(i);
        const auto& result_vector = result.at(i);
        ASSERT_THAT(result_vector, ::testing::SizeIs(expected_vector.size()));
        for (size_t j = 0; j < expected_vector.size(); ++j) {
            ASSERT_THAT(result_vector.at(j), ::testing::DoubleEq(expected_vector.at(j)));
        }

        CHECK_CUDA_ERR(cudaFree(d_burst_indices.array));
        CHECK_CUDA_ERR(cudaFree(d_sample_indices.array));
        alus::tnr::device::DestroyBurstIndexToInterpolatedRangeVectorMap(d_map);
    }

    for (auto& vector : h_burst_to_range_index_map) {
        CHECK_CUDA_ERR(cudaFree(vector.noise_lut.array));
        CHECK_CUDA_ERR(cudaFree(vector.pixels.array));
    }

    CHECK_CUDA_ERR(cudaFree(d_burst_index_to_range_vector_map.array));
}

TEST_F(ThermalNoiseKernelTest, CalculateNoiseMatrixTest) {
    // TEST INPUT CONSTANTS
    const std::vector<alus::Rectangle> tiles{{2140, 0, 10, 10}, {2140, 1780, 10, 10}};
    const int lines_per_burst{1503};
    const std::vector<std::vector<double>> interpolated_azimuth_vectors{
        {1.1698060035705566, 1.1693260073661804, 1.1688460111618042, 1.168366014957428, 1.1678860187530518,
         1.1674060225486755, 1.1669260263442993, 1.166446030139923, 1.1659660339355469, 1.1654860377311707},
        {1.065335202217102, 1.065060806274414, 1.0647864103317262, 1.064512014389038, 1.0642383098602295,
         1.063964605331421, 1.0636909008026123, 1.0634171962738037, 1.0631434917449951,
         1.0628697872161865}};  // One vector per tile
    const std::vector<std::vector<double>> interpolated_range_vectors{
        {434.6820983886719, 434.6494537353516, 434.61680908203124, 434.58416442871095, 434.5515197753906,
         434.5188751220703, 434.48623046875, 434.4535858154297, 434.4209411621094, 434.38829650878904},
        {430.21739196777344, 430.18510208129885, 430.1528121948242, 430.1205223083496, 430.088232421875,
         430.0559425354004, 430.0236526489258, 429.99136276245116, 429.9590728759766,
         429.92678298950193}};  // One vector per burst. Only two bursts.

    // COPY DATA TO GPU
    std::vector<alus::cuda::KernelArray<double>> d_azimuth_vectors{interpolated_azimuth_vectors.size()};
    for (size_t i = 0; i < interpolated_azimuth_vectors.size(); ++i) {
        const auto& h_vector = interpolated_azimuth_vectors.at(i);
        auto& d_vector = d_azimuth_vectors.at(i);
        d_vector.size = h_vector.size();
        CHECK_CUDA_ERR(cudaMalloc(&d_vector.array, d_vector.ByteSize()));
        CHECK_CUDA_ERR(cudaMemcpy(d_vector.array, h_vector.data(), d_vector.ByteSize(), cudaMemcpyHostToDevice));
    }
    auto burst_to_range_vector_map =
        alus::tnr::device::CopyBurstIndexToInterpolatedRangeVectorMapToDevice(interpolated_range_vectors);

    // ACTUAL TEST BODY
    ASSERT_THAT(tiles, ::testing::SizeIs(test::expectedvalues::NOISE_MATRICES.size()));
    ASSERT_THAT(interpolated_azimuth_vectors, ::testing::SizeIs(test::expectedvalues::NOISE_MATRICES.size()));
    ASSERT_THAT(interpolated_range_vectors, ::testing::SizeIs(test::expectedvalues::NOISE_MATRICES.size()));

    for (size_t i = 0; i < test::expectedvalues::NOISE_MATRICES.size(); ++i) {
        const auto calculated_noise_matrix = tnr::CalculateNoiseMatrix(
            tiles.at(i), lines_per_burst, d_azimuth_vectors.at(i), burst_to_range_vector_map, stream_);

        const auto& expected_matrix = test::expectedvalues::NOISE_MATRICES.at(i);
        const auto h_matrix = alus::tnr::device::CopyMatrixToHost(calculated_noise_matrix);

        ASSERT_THAT(h_matrix, ::testing::SizeIs(expected_matrix.size()));
        for (size_t j = 0; j < expected_matrix.size(); ++j) {
            const auto& expected_row = expected_matrix.at(j);
            const auto& calculated_row = h_matrix.at(j);
            ASSERT_THAT(calculated_row, ::testing::SizeIs(expected_row.size()));
            for (size_t k = 0; k < expected_row.size(); ++k) {
                ASSERT_THAT(calculated_row.at(k), ::testing::DoubleEq(expected_row.at(k)));
            }
        }

        tnr::device::DestroyKernelMatrix(calculated_noise_matrix);
    }

    for (auto& vector : d_azimuth_vectors) {
        CHECK_CUDA_ERR(cudaFree(vector.array));
    }
    alus::tnr::device::DestroyBurstIndexToInterpolatedRangeVectorMap(burst_to_range_vector_map);
}
}  // namespace