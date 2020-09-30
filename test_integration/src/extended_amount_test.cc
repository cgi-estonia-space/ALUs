#include "gmock/gmock.h"

#include "../goods/compute_extended_amount_test_data.h"
#include "backgeocoding.h"
#include "extended_amount_computation.h"

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
    EXPECT_THAT(result.range_max, testing::DoubleNear(expected_output.range_max, 1e-9));
    EXPECT_THAT(result.range_min, testing::DoubleNear(expected_output.range_min, 1e-9));
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
    for (auto && pair : alus::goods::GET_BURST_INDEX_VALUES) {
        auto result = alus::backgeocoding::GetBurstIndex(pair.first, NUMBER_OF_BURSTS, LINES_PER_BURST);
        EXPECT_THAT(result, testing::Eq(pair.second));
    }
}
}  // namespace