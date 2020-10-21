#include "gmock/gmock.h"

#include "../goods/compute_extended_amount_test_data.h"
#include "backgeocoding.h"
#include "extended_amount.h"

#include "backgeocoding_utils.cuh"

namespace {

TEST(ExtendedAmountTest, ComputeExtendedAmount) {
    alus::backgeocoding::Backgeocoding backgeocoding;

    backgeocoding.SetSentinel1Placeholders("./goods/backgeocoding/dcEstimateList.txt",
                                           "./goods/backgeocoding/azimuthList.txt",
                                           "./goods/backgeocoding/masterBurstLineTimes.txt",
                                           "./goods/backgeocoding/slaveBurstLineTimes.txt",
                                           "./goods/backgeocoding/masterGeoLocation.txt",
                                           "./goods/backgeocoding/slaveGeoLocation.txt");

    backgeocoding.SetOrbitVectorsFiles("./goods/backgeocoding/masterOrbitStateVectors.txt",
                                       "./goods/backgeocoding/slaveOrbitStateVectors.txt");
    backgeocoding.SetSRTMDirectory("./goods/");
    backgeocoding.SetEGMGridFile("./goods/backgeocoding/ww15mgh_b.grd");
    backgeocoding.FeedPlaceHolders();
    backgeocoding.PrepareToCompute();

    alus::Rectangle const input{4000, 17000, 100, 100};
    alus::backgeocoding::AzimuthAndRangeBounds const expected_output{
        -0.01773467106249882, 0.0, -3.770974349203243, 3.8862058607542167};

    alus::backgeocoding::AzimuthAndRangeBounds result =
        backgeocoding.ComputeExtendedAmount(input.x, input.y, input.width, input.height);
    EXPECT_THAT(result.azimuth_min, testing::DoubleNear(expected_output.azimuth_min, 0.01));
    EXPECT_THAT(result.azimuth_max, testing::DoubleNear(expected_output.azimuth_max, 0.01));
    EXPECT_THAT(result.range_max, testing::DoubleNear(expected_output.range_max, 0.01));
    EXPECT_THAT(result.range_min, testing::DoubleNear(expected_output.range_min, 0.01));
}

TEST(ExtendedAmountTest, GetBurstIndexTest) {
    int const NUMBER_OF_BURSTS{19};
    int const LINES_PER_BURST{1503};
    std::for_each(
        alus::goods::GET_BURST_INDEX_VALUES.begin(), alus::goods::GET_BURST_INDEX_VALUES.end(), [](auto pair) {
            auto result = alus::backgeocoding::GetBurstIndex(pair.first, NUMBER_OF_BURSTS, LINES_PER_BURST);
            EXPECT_THAT(result, testing::Eq(pair.second));
        });
}
}  // namespace