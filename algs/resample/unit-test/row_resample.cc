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

#include "row_resample.h"

#include <array>

#include "gmock/gmock.h"

#include "cuda_ptr.h"

namespace {

using ::testing::DoubleNear;
using ::testing::FloatNear;
using ::testing::Pointwise;

TEST(RowResample, GetRatioCalculatesCorrently) {
    std::array<int, 6> input_len{6, 13, 4, 1013, 2934, 20123};
    std::array<int, 6> output_len{10, 55, 5, 2087, 10894, 89456};
    std::array<double, 6> expected{
        0.6, 0.23636363636363636, 0.8, 0.4853857211308098, 0.2693225628786488, 0.22494857807190127};

    for (size_t i{}; i < input_len.size(); i++) {
        ASSERT_THAT(alus::rowresample::GetRatio(input_len.at(i), output_len.at(i)), DoubleNear(expected.at(i), 1e-20))
            << "Occurred at element " << i;
    }
}

TEST(RowResample, FillLineFromCalculatesCorrectly) {
    std::array<float, 6> input{10, 15, 15, 0, 0, 1};
    std::array<float, 10> expected_out{10, 11.66666, 15, 15, 15, 0, 0, 0, 0.6666, 1};

    std::array<float, 10> output{};
    alus::rowresample::FillLineFrom(input.data(), input.size(), output.data(), output.size());
    ASSERT_THAT(output, Pointwise(FloatNear(1e-3), expected_out));
}

TEST(RowResample, ProcessReturnsCorrectResults) {
    {
        std::array<float, 12> input{10, 15, 15, 0, 0, 1, 10, 15, 15, 0, 0, 1};
        std::array<float, 20> expected_out{10, 11.66666, 15, 15, 15, 0, 0, 0, 0.6666, 1,
                                           10, 11.66666, 15, 15, 15, 0, 0, 0, 0.6666, 1};

        alus::cuda::CudaPtr<float> dev_in(input.size());
        CHECK_CUDA_ERR(cudaMemcpy(dev_in.Get(), input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice));
        alus::cuda::CudaPtr<float> dev_out(expected_out.size());

        alus::rowresample::Process(dev_in.Get(), {6, 2}, dev_out.Get(), {10, 2});
        std::array<float, expected_out.size()> results;
        CHECK_CUDA_ERR(
            cudaMemcpy(results.data(), dev_out.Get(), results.size() * sizeof(float), cudaMemcpyDeviceToHost));

        ASSERT_THAT(results, Pointwise(FloatNear(1e-3), expected_out));
    }

    {
        // clang-format off
        std::array<float, 12> input{10, 15, 15, 0, 0, 1,
                                    10, 15, 15, 0, 0, 1};
        std::array<float, 24> expected{10, 10, 15, 15, 15, 15, 0, 0, 0, 0, 1, 1,
                                       10, 10, 15, 15, 15, 15, 0, 0, 0, 0, 1, 1};
        // clang-format on

        alus::cuda::CudaPtr<float> dev_in(input.size());
        CHECK_CUDA_ERR(cudaMemcpy(dev_in.Get(), input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice));
        alus::cuda::CudaPtr<float> dev_out(expected.size());

        alus::rowresample::Process(dev_in.Get(), {6, 2}, dev_out.Get(), {12, 2});
        std::array<float, expected.size()> results;
        CHECK_CUDA_ERR(
            cudaMemcpy(results.data(), dev_out.Get(), results.size() * sizeof(float), cudaMemcpyDeviceToHost));

        ASSERT_THAT(results, Pointwise(FloatNear(1e-3), expected));
    }
}

TEST(RowResample, ProcessAndTransferHostReturnsCorrectResults) {
    {
        std::array<float, 12> input{10, 15, 15, 0, 0, 1, 10, 15, 15, 0, 0, 1};
        std::array<float, 20> expected_out{10, 11.66666, 15, 15, 15, 0, 0, 0, 0.6666, 1,
                                           10, 11.66666, 15, 15, 15, 0, 0, 0, 0.6666, 1};

        std::array<float, expected_out.size()> results;
        alus::rowresample::ProcessAndTransferHost(input.data(), {6, 2}, results.data(), {10, 2});

        ASSERT_THAT(results, Pointwise(FloatNear(1e-3), expected_out));
    }

    {
        // clang-format off
        std::array<float, 12> input{10, 15, 15, 0, 0, 1,
                                    10, 15, 15, 0, 0, 1};
        std::array<float, 24> expected{10, 10, 15, 15, 15, 15, 0, 0, 0, 0, 1, 1,
                                       10, 10, 15, 15, 15, 15, 0, 0, 0, 0, 1, 1};
        // clang-format on

        std::array<float, expected.size()> results;
        alus::rowresample::ProcessAndTransferHost(input.data(), {6, 2}, results.data(), {12, 2});

        ASSERT_THAT(results, Pointwise(FloatNear(1e-3), expected));
    }
}

}  // namespace
