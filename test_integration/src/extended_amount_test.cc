#include "gmock/gmock.h"

#include "../goods/compute_extended_amount_test_data.h"
#include "backgeocoding.h"
#include "extended_amount_computation.h"

#include "backgeocoding_utils.cuh"

namespace {

TEST(ExtendedAmountTest, ComputeExtendedAmount) {
    alus::backgeocoding::Backgeocoding backgeocoding;

    backgeocoding.SetSRTMDirectory("./goods/");
    backgeocoding.SetEGMGridFile("./goods/backgeocoding/ww15mgh_b.grd");
    backgeocoding.FeedPlaceHolders();
    backgeocoding.PrepareToCompute("./goods/master_metadata.dim", "./goods/slave_metadata.dim");

    alus::Rectangle const input{4000, 17000, 100, 100};
    alus::backgeocoding::AzimuthAndRangeBounds const expected_output{
        -0.01785065843796474, 0.0, -3.185590399236844, 3.8288619352210844};

    alus::backgeocoding::AzimuthAndRangeBounds result =
        backgeocoding.ComputeExtendedAmount(input.x, input.y, input.width, input.height);
    EXPECT_THAT(result.range_max, testing::DoubleNear(expected_output.range_max, 1e-6));
    EXPECT_THAT(result.range_min, testing::DoubleNear(expected_output.range_min, 1e-6));
    EXPECT_THAT(result.azimuth_min, testing::DoubleEq(expected_output.azimuth_min));
    EXPECT_THAT(result.azimuth_max, testing::DoubleEq(expected_output.azimuth_max));


    alus::Rectangle const input_irregular{10, 19486, 50, 50};
    alus::backgeocoding::AzimuthAndRangeBounds const expected_irregular_output{
        -0.006566788426425774, 0.003858104329992784, 0.0, 17.145334103861202};

    alus::backgeocoding::AzimuthAndRangeBounds irregular_result =
        backgeocoding.ComputeExtendedAmount(input_irregular.x, input_irregular.y, input_irregular.width, input_irregular.height);

    EXPECT_THAT(irregular_result.range_max, testing::DoubleNear(expected_irregular_output.range_max, 1e-9));
    EXPECT_THAT(irregular_result.range_min, testing::DoubleNear(expected_irregular_output.range_min, 1e-9));
    EXPECT_THAT(irregular_result.azimuth_min, testing::DoubleEq(expected_irregular_output.azimuth_min));
    EXPECT_THAT(irregular_result.azimuth_max, testing::DoubleEq(expected_irregular_output.azimuth_max));
}

TEST(ExtendedAmountTest, GetBurstIndexTest) {
    int const NUMBER_OF_BURSTS{19};
    int const LINES_PER_BURST{1503};
    (void)NUMBER_OF_BURSTS;
    (void)LINES_PER_BURST;
    
    std::for_each(
        alus::goods::GET_BURST_INDEX_VALUES.begin(), alus::goods::GET_BURST_INDEX_VALUES.end(), [](auto pair) {
            auto result = alus::backgeocoding::GetBurstIndex(pair.first, NUMBER_OF_BURSTS, LINES_PER_BURST);
            EXPECT_THAT(result, testing::Eq(pair.second));
        });
}
}  // namespace
