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

#include <map>

#include "shapes.h"
#include "topsar_merge.h"

namespace {
using namespace alus;
using namespace alus::topsarmerge;

class TopsarMergeTest : public ::testing::Test {
protected:
    const MergeOperatorParameters one_swath_operator_parameters_{3,
                                                                 "IW",
                                                                 "SLC",
                                                                 1,
                                                                 6.16003436348297E8,
                                                                 6.16003446880967E8,
                                                                 0.002055556299999998,
                                                                 0.002681103475533822,
                                                                 0.00320170920183382,
                                                                 7.770582407379975E-9,
                                                                 5123,
                                                                 66997};

    const std::map<size_t, SubSwathMergeInfo> one_sub_swath_merge_info_ = {
        {0,
         {0.002681103475533822, 0.002839203745194375, 6.160034363914636E8, 6.160034393987426E8, 6.16003436348297E8,
          6.16003439433687E8, 0.002055556299999998, 0.002668787102418125, 0.0028466557337230524, 22891, 1502,
          2.329562}},
        {1,
         {0.002834875538419279, 0.0030237473144130566, 6.16003440105854E8, 6.160034431049106E8, 6.16003440050354E8,
          6.16003443158355E8, 0.002055556299999998, 0.002823647046840615, 0.0030308030032389576, 26660, 1513,
          2.329562}},
        {2,
         {0.0030188984801177987, 0.00320170920183382, 6.160034438223001E8, 6.160034468254678E8, 6.16003443762689E8,
          6.16003446880967E8, 0.002055556299999998, 0.0030089909875483894, 0.003208081079407872, 25622, 1518,
          2.329562}}};

    const MergeOperatorParameters multiple_swath_operator_parameters_{2,
                                                                      "IW",
                                                                      "SLC",
                                                                      1,
                                                                      6.16003436348297E8,
                                                                      6.16003440399798E8,
                                                                      0.002055556299999998,
                                                                      0.002681103475533822,
                                                                      0.0030237473144130566,
                                                                      7.770582407379975E-9,
                                                                      1970,
                                                                      44095};

    const std::map<size_t, SubSwathMergeInfo> multiple_sub_swath_merge_info_ = {
        {0,
         {0.002681103475533822, 0.002839203745194375, 6.160034363914636E8, 6.160034393987426E8, 6.16003436348297E8,
          6.16003439433687E8, 0.002055556299999998, 0.002668787102418125, 0.0028466557337230524, 22891, 1502,
          2.329562}},
        {1,
         {0.002823647046840615, 0.0030308030032389576, 6.160034373452414E8, 6.160034403484092E8, 6.16003437291796E8,
          6.16003440399798E8, 0.002055556299999998, 0.002823647046840615, 0.0030308030032389576, 26660, 1513,
          2.329562}}};

    static constexpr std::string_view SUBSWATH_1_NAME{"IW1"};
    static constexpr std::string_view SUBSWATH_2_NAME{"IW2"};
    static constexpr std::string_view SUBSWATH_3_NAME{"IW3"};
};

TEST_F(TopsarMergeTest, FindFirstAndLastSubSwathIndicesTest) {
    const Rectangle target_rectangle_1{2680, 0, 536, 428};
    const int expected_first_index_1{0};
    const int expected_last_index_1{0};

    int first_index{-1};
    int last_index{-1};
    FindFirstAndLastSubSwathIndices(first_index, last_index, target_rectangle_1, one_swath_operator_parameters_,
                                    one_sub_swath_merge_info_);
    ASSERT_THAT(first_index, ::testing::Eq(expected_first_index_1));
    ASSERT_THAT(last_index, ::testing::Eq(expected_last_index_1));

    first_index = -1;
    last_index = -1;

    const Rectangle target_rectangle_2{19296, 1712, 536, 428};
    const int expected_first_index_2{1};
    const int expected_last_index_2{1};

    FindFirstAndLastSubSwathIndices(first_index, last_index, target_rectangle_2, one_swath_operator_parameters_,
                                    one_sub_swath_merge_info_);
    ASSERT_THAT(first_index, ::testing::Eq(expected_first_index_2));
    ASSERT_THAT(last_index, ::testing::Eq(expected_last_index_2));
}

TEST_F(TopsarMergeTest, ComputeTargetStartEndTimeTest) {
    MergeOperatorParameters computed_operator_parameters{};
    computed_operator_parameters.number_of_subswaths = 3;
    ComputeTargetStartEndTime(computed_operator_parameters, one_sub_swath_merge_info_);

    ASSERT_THAT(computed_operator_parameters.target_first_line_time,
                ::testing::DoubleEq(one_swath_operator_parameters_.target_first_line_time));
    ASSERT_THAT(computed_operator_parameters.target_last_line_time,
                ::testing::DoubleEq(one_swath_operator_parameters_.target_last_line_time));
    ASSERT_THAT(computed_operator_parameters.target_line_time_interval,
                ::testing::DoubleEq(one_swath_operator_parameters_.target_line_time_interval));
}

TEST_F(TopsarMergeTest, GetSubSwathIndexFromNameTest) {
    const int sub_swath_1_index = GetSubSwathIndexFromName(SUBSWATH_1_NAME);
    const int sub_swath_2_index = GetSubSwathIndexFromName(SUBSWATH_2_NAME);
    const int sub_swath_3_index = GetSubSwathIndexFromName(SUBSWATH_3_NAME);

    const int expected_sub_swath_1_index{1};
    const int expected_sub_swath_2_index{2};
    const int expected_sub_swath_3_index{3};

    ASSERT_THAT(sub_swath_1_index, ::testing::Eq(expected_sub_swath_1_index));
    ASSERT_THAT(sub_swath_2_index, ::testing::Eq(expected_sub_swath_2_index));
    ASSERT_THAT(sub_swath_3_index, ::testing::Eq(expected_sub_swath_3_index));
}

TEST_F(TopsarMergeTest, GetSubSwathIndexTest) {
    const int x_1{19928};
    const int y_1{396};
    const int first_sub_swath_index{0};
    const int last_sub_swath_index{1};
    const int expected_sub_swath_index_1{0};

    const auto actual_sub_swath_index_1 =
        GetSubSwathIndex(x_1, y_1, first_sub_swath_index, last_sub_swath_index, multiple_swath_operator_parameters_,
                         multiple_sub_swath_merge_info_);

    ASSERT_THAT(actual_sub_swath_index_1, ::testing::Eq(expected_sub_swath_index_1));

    const int x_2{20347};
    const int y_2{396};
    const int expected_sub_swath_index_2{-1};

    const auto actual_sub_swath_index_2 =
        GetSubSwathIndex(x_2, y_2, first_sub_swath_index, last_sub_swath_index, multiple_swath_operator_parameters_,
                         multiple_sub_swath_merge_info_);

    ASSERT_THAT(actual_sub_swath_index_2, ::testing::Eq(expected_sub_swath_index_2));
}

TEST_F(TopsarMergeTest, GetSubSwathIndexBySlrTimeTest) {
    const double slr_1{0.002681103475533822};
    const int expected_sub_swath_index_1{0};

    const auto actual_sub_swath_index_1 =
        GetSubSwathIndexBySlrTime(slr_1, multiple_swath_operator_parameters_, multiple_sub_swath_merge_info_);

    ASSERT_THAT(actual_sub_swath_index_1, ::testing::Eq(expected_sub_swath_index_1));

    const double slr_2{0.002863269238910031};
    const int expected_sub_swath_index_2{1};

    const auto actual_sub_swath_index_2 =
        GetSubSwathIndexBySlrTime(slr_2, one_swath_operator_parameters_, one_sub_swath_merge_info_);

    ASSERT_THAT(actual_sub_swath_index_2, ::testing::Eq(expected_sub_swath_index_2));
}

TEST_F(TopsarMergeTest, GetTargetBandNameFromSourceBandNameTest) {
    const std::string_view source_name_1{"i_IW1_VH"};
    const std::string_view expected_name_1{"i_VH"};

    const auto target_name_1 =
        GetTargetBandNameFromSourceBandName(source_name_1, one_swath_operator_parameters_.acquisition_mode);
    ASSERT_THAT(target_name_1.data(), ::testing::StrEq(expected_name_1.data()));

    const std::string_view source_name_2{"q_IW1_VH"};
    const std::string_view expected_name_2{"q_VH"};

    const auto target_name_2 =
        GetTargetBandNameFromSourceBandName(source_name_2, one_swath_operator_parameters_.acquisition_mode);
    ASSERT_THAT(target_name_2.data(), ::testing::StrEq(expected_name_2.data()));

    const std::string_view source_name_3{"i_IW2_VH"};
    const std::string_view expected_name_3{"i_VH"};

    const auto target_name_3 =
        GetTargetBandNameFromSourceBandName(source_name_3, one_swath_operator_parameters_.acquisition_mode);
    ASSERT_THAT(target_name_3.data(), ::testing::StrEq(expected_name_3.data()));
}

TEST_F(TopsarMergeTest, GetSampleIndexInSourceProductTest) {
    const int x_1{2144};
    const int expected_index_1{3729};

    const int index_1 = GetSampleIndexInSourceProduct(x_1, one_sub_swath_merge_info_.at(0).number_of_samples,
                                                      one_sub_swath_merge_info_.at(0).slr_time_to_first_pixel,
                                                      one_swath_operator_parameters_);
    ASSERT_THAT(index_1, ::testing::Eq(expected_index_1));

    const int x_2{2679};
    const int expected_index_2{4264};
    const int index_2 = GetSampleIndexInSourceProduct(x_2, one_sub_swath_merge_info_.at(0).number_of_samples,
                                                      one_sub_swath_merge_info_.at(0).slr_time_to_first_pixel,
                                                      one_swath_operator_parameters_);
    ASSERT_THAT(index_2, ::testing::Eq(expected_index_2));
}

TEST_F(TopsarMergeTest, GetLineIndexInSourceProductTest) {
    const int y_1{1712};
    const int expected_index_1{0};

    const auto index_1 =
        GetLineIndexInSourceProduct(y_1, one_sub_swath_merge_info_.at(1), one_swath_operator_parameters_);
    ASSERT_THAT(index_1, ::testing::Eq(expected_index_1));

    const int y_2{2139};
    const int expected_index_2{338};

    const auto index_2 =
        GetLineIndexInSourceProduct(y_2, one_sub_swath_merge_info_.at(1), one_swath_operator_parameters_);
    ASSERT_THAT(index_2, ::testing::Eq(expected_index_2));
}

TEST_F(TopsarMergeTest, GetSourceRectanglesTest) {
    const int number_of_source_tiles_1{2};
    const int first_sub_swath_index_1{0};
    const int last_sub_swath_index_1{1};
    const Rectangle target_rectangle_1{19504, 396, 424, 396};
    const std::vector<Rectangle> expected_rectangles_1 = {{21089, 396, 424, 396}, {1160, 0, 424, 333}};

    const auto calculated_rectangles_1 =
        GetSourceRectangles(number_of_source_tiles_1, first_sub_swath_index_1, last_sub_swath_index_1,
                            target_rectangle_1, multiple_swath_operator_parameters_, multiple_sub_swath_merge_info_);

    ASSERT_THAT(calculated_rectangles_1.size(), ::testing::Eq(expected_rectangles_1.size()));
    for (size_t i = 0; i < expected_rectangles_1.size(); ++i) {
        const auto& expected_rectangle = expected_rectangles_1.at(i);
        const auto& calculated_rectangle = calculated_rectangles_1.at(i);
        ASSERT_THAT(calculated_rectangle.x, ::testing::Eq(expected_rectangle.x));
        ASSERT_THAT(calculated_rectangle.y, ::testing::Eq(expected_rectangle.y));
        ASSERT_THAT(calculated_rectangle.width, ::testing::Eq(expected_rectangle.width));
        ASSERT_THAT(calculated_rectangle.height, ::testing::Eq(expected_rectangle.height));
    }

    const int number_of_source_tiles_2{1};
    const int first_sub_swath_index_2{0};
    const int last_sub_swath_index_2{0};
    const Rectangle target_rectangle_2{1072, 0, 536, 428};
    const std::vector<Rectangle> expected_rectangles_2 = {{2657, 0, 536, 428}};

    const auto calculated_rectangles_2 =
        GetSourceRectangles(number_of_source_tiles_2, first_sub_swath_index_2, last_sub_swath_index_2,
                            target_rectangle_2, one_swath_operator_parameters_, one_sub_swath_merge_info_);

    ASSERT_THAT(calculated_rectangles_2.size(), ::testing::Eq(expected_rectangles_2.size()));
    for (size_t i = 0; i < expected_rectangles_2.size(); ++i) {
        const auto& expected_rectangle = expected_rectangles_2.at(i);
        const auto& calculated_rectangle = calculated_rectangles_2.at(i);
        ASSERT_THAT(calculated_rectangle.x, ::testing::Eq(expected_rectangle.x));
        ASSERT_THAT(calculated_rectangle.y, ::testing::Eq(expected_rectangle.y));
        ASSERT_THAT(calculated_rectangle.width, ::testing::Eq(expected_rectangle.width));
        ASSERT_THAT(calculated_rectangle.height, ::testing::Eq(expected_rectangle.height));
    }
}

TEST_F(TopsarMergeTest, ComputeTargetSlantRangeTimeToFirstAndLastPixelsTest) {
    MergeOperatorParameters parameters{};
    parameters.number_of_subswaths = 3;

    ComputeTargetSlantRangeTimeToFirstAndLastPixels(parameters, one_sub_swath_merge_info_);
    ASSERT_THAT(parameters.target_slant_range_time_to_first_pixel,
                ::testing::DoubleEq(one_swath_operator_parameters_.target_slant_range_time_to_first_pixel));
    ASSERT_THAT(parameters.target_slant_range_time_to_last_pixel,
                ::testing::DoubleEq(one_swath_operator_parameters_.target_slant_range_time_to_last_pixel));
    ASSERT_THAT(parameters.target_delta_slant_range_time,
                ::testing::DoubleEq(one_swath_operator_parameters_.target_delta_slant_range_time));
}

TEST_F(TopsarMergeTest, ComputeTargetWidthAndHeightTest) {
    MergeOperatorParameters parameters_1 = one_swath_operator_parameters_;
    parameters_1.target_width = 0;
    parameters_1.target_height = 0;

    ComputeTargetWidthAndHeight(parameters_1);
    ASSERT_THAT(parameters_1.target_width, ::testing::Eq(one_swath_operator_parameters_.target_width));
    ASSERT_THAT(parameters_1.target_height, ::testing::Eq(one_swath_operator_parameters_.target_height));

    MergeOperatorParameters parameters_2 = multiple_swath_operator_parameters_;
    parameters_2.target_width = 0;
    parameters_2.target_height = 0;

    ComputeTargetWidthAndHeight(parameters_2);
    ASSERT_THAT(parameters_2.target_width, ::testing::Eq(multiple_swath_operator_parameters_.target_width));
    ASSERT_THAT(parameters_2.target_height, ::testing::Eq(multiple_swath_operator_parameters_.target_height));
}
}  // namespace
