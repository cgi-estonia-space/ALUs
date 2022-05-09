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

#include "../goods/compute_extended_amount_test_data.h"
#include "backgeocoding.h"
#include "dem_assistant.h"
#include "extended_amount_computation.h"

#include "backgeocoding_utils.cuh"

namespace {

TEST(ExtendedAmountTest, ComputeExtendedAmount) {
    std::vector<std::string> srtm3_files{"./goods/srtm_41_01.tif", "./goods/srtm_41_01.tif"};
    std::shared_ptr<alus::app::DemAssistant> dem_assistant =
        alus::app::DemAssistant::CreateFormattedSrtm3TilesOnGpuFrom(std::move(srtm3_files));
    alus::backgeocoding::Backgeocoding backgeocoding;

    dem_assistant->GetSrtm3Manager()->HostToDevice();
    backgeocoding.SetElevationData(dem_assistant->GetEgm96Manager()->GetDeviceValues(),
                                   {dem_assistant->GetSrtm3Manager()->GetSrtmBuffersInfo(),
                                    dem_assistant->GetSrtm3Manager()->GetDeviceSrtm3TilesCount()},
                                   true);
    backgeocoding.PrepareToCompute("./goods/master_metadata.dim", "./goods/slave_metadata.dim");

    alus::Rectangle const input{4000, 17000, 100, 100};
    alus::backgeocoding::AzimuthAndRangeBounds const expected_output{0, 0, -3, 3};

    alus::backgeocoding::AzimuthAndRangeBounds result =
        backgeocoding.ComputeExtendedAmount(input.x, input.y, input.width, input.height);
    EXPECT_EQ(result.range_max, expected_output.range_max);
    EXPECT_EQ(result.range_min, expected_output.range_min);
    EXPECT_EQ(result.azimuth_min, expected_output.azimuth_min);
    EXPECT_EQ(result.azimuth_max, expected_output.azimuth_max);

    alus::Rectangle const input_irregular{10, 19486, 50, 50};
    alus::backgeocoding::AzimuthAndRangeBounds const expected_irregular_output{0, 0, 0, 17};

    alus::backgeocoding::AzimuthAndRangeBounds irregular_result = backgeocoding.ComputeExtendedAmount(
        input_irregular.x, input_irregular.y, input_irregular.width, input_irregular.height);

    EXPECT_EQ(irregular_result.range_max, expected_irregular_output.range_max);
    EXPECT_EQ(irregular_result.range_min, expected_irregular_output.range_min);
    EXPECT_EQ(irregular_result.azimuth_min, expected_irregular_output.azimuth_min);
    EXPECT_EQ(irregular_result.azimuth_max, expected_irregular_output.azimuth_max);
}

TEST(ExtendedAmountTest, GetBurstIndexTest) {
    int const number_of_bursts{19};
    int const lines_per_burst{1503};
    (void)number_of_bursts;
    (void)lines_per_burst;

    std::for_each(alus::goods::GET_BURST_INDEX_VALUES.begin(), alus::goods::GET_BURST_INDEX_VALUES.end(),
                  [](auto pair) {
                      auto result = alus::backgeocoding::GetBurstIndex(pair.first, number_of_bursts, lines_per_burst);
                      EXPECT_THAT(result, testing::Eq(pair.second));
                  });
}
}  // namespace
