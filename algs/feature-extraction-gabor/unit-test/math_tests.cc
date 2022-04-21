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

#include <vector>

#include "gmock/gmock.h"

#include "cuda_copies.h"
#include "cuda_ptr.h"
#include "patch_reduction.h"

namespace {

using ::testing::FloatNear;

namespace {
constexpr size_t PATCH_SIZE = 100;
constexpr size_t EDGE_SIZE = 27;
constexpr size_t PATCH_PADDED_SIZE = PATCH_SIZE + EDGE_SIZE;
constexpr size_t X_PATCHES = 2;
constexpr size_t Y_PATCHES = 2;
constexpr size_t WIDTH = X_PATCHES * PATCH_PADDED_SIZE;
constexpr size_t HEIGHT = Y_PATCHES * PATCH_PADDED_SIZE;

const std::vector<float> EXPECTED_MEANS = {1.0F, 0.0F, 0.5F, 5000.5F};
std::vector<float> GenerateTestBuffer() {
    std::vector<float> vec(WIDTH * HEIGHT);

    const size_t pad_size = EDGE_SIZE / 2;

    // (0,0)
    {
        for (size_t y = 0; y < PATCH_SIZE; y++) {
            for (size_t x = 0; x < PATCH_SIZE; x++) {
                size_t idx = (pad_size + y) * WIDTH + (pad_size + x);
                vec.at(idx) = 1.0F;
            }
        }
    }
    // (1, 0)
    //(0,1)
    {
        for (size_t y = 0; y < PATCH_SIZE; y++) {
            for (size_t x = 0; x < PATCH_SIZE; x++) {
                size_t idx = (pad_size + y + PATCH_PADDED_SIZE) * WIDTH + (pad_size + x);
                vec.at(idx) = (x & 1) ? 1.0F : 0.0F;
            }
        }
    }

    //(1,1)
    {
        float cnt = 1.0F;
        for (size_t y = 0; y < PATCH_SIZE; y++) {
            for (size_t x = 0; x < PATCH_SIZE; x++) {
                size_t idx = (pad_size + y + PATCH_PADDED_SIZE) * WIDTH + (x + PATCH_PADDED_SIZE + pad_size);
                vec.at(idx) = cnt++;
            }
        }
    }
    return vec;
}
}  // namespace

TEST(FilterBank, MeanCalculatedCorrectly) {
    const std::vector<float> h_data = GenerateTestBuffer();

    alus::cuda::DeviceBuffer<float> d_data(h_data.size());
    alus::cuda::CopyArrayH2D(d_data.Get(), h_data.data(), h_data.size());

    alus::cuda::DeviceBuffer<float> d_means(X_PATCHES * Y_PATCHES);
    alus::featurextractiongabor::LaunchPatchMeanReduction(d_data.Get(), d_means.Get(), PATCH_SIZE, EDGE_SIZE, X_PATCHES,
                                                          Y_PATCHES);

    std::vector<float> h_means(d_means.size());
    alus::cuda::CopyArrayD2H(h_means.data(), d_means.Get(), h_means.size());

    for (size_t i = 0; i < h_means.size(); i++) {
        ASSERT_EQ(h_means.at(i), EXPECTED_MEANS.at(i));
    }
}

TEST(FilterBank, StdDev) {
    const std::vector<float> h_data = GenerateTestBuffer();

    alus::cuda::DeviceBuffer<float> d_data(h_data.size());
    alus::cuda::CopyArrayH2D(d_data.Get(), h_data.data(), h_data.size());

    alus::cuda::DeviceBuffer<float> d_means(X_PATCHES * Y_PATCHES);
    alus::cuda::DeviceBuffer<float> d_std_devs(X_PATCHES * Y_PATCHES);
    alus::cuda::CopyArrayH2D(d_means.Get(), EXPECTED_MEANS.data(), d_means.size());
    alus::featurextractiongabor::LaunchPatchStdDevReduction(d_data.Get(), d_means.Get(), d_std_devs.Get(), PATCH_SIZE,
                                                            EDGE_SIZE, X_PATCHES, Y_PATCHES);

    std::vector<float> h_std_devs(d_std_devs.size());
    alus::cuda::CopyArrayD2H(h_std_devs.data(), d_std_devs.Get(), h_std_devs.size());

    EXPECT_EQ(h_std_devs.at(0), 0.0F);
    EXPECT_EQ(h_std_devs.at(1), 0.0F);
    ASSERT_THAT(h_std_devs.at(2), FloatNear(0.5F, 0.001F));
    ASSERT_THAT(h_std_devs.at(3), FloatNear(2886.9F, 0.1F));
}
}  // namespace