#include <cstddef>
#include <vector>

#include "gmock/gmock.h"

#include "../include/patch_assembly.h"

// Symbols from linked patch object files
extern char _binary_patch_input_band1_0_0_50_50_img_start[];
extern char _binary_patch_input_band1_0_0_50_50_img_end[];
extern char _binary_patch_padded_band1_0_0_77_77_img_start[];
extern char _binary_patch_padded_band1_0_0_77_77_img_end[];

namespace {

using ::testing::Eq;
using ::testing::FloatEq;

class PatchFullAssembly : public ::testing::Test {
public:
    const uint8_t* input_patch_data_ = reinterpret_cast<uint8_t*>(_binary_patch_input_band1_0_0_50_50_img_start);
    const float* expected_padded_patch_data_ = reinterpret_cast<float*>(_binary_patch_padded_band1_0_0_77_77_img_start);
    const size_t expected_padded_patch_size_ =
        _binary_patch_padded_band1_0_0_77_77_img_end - _binary_patch_padded_band1_0_0_77_77_img_start;

    static constexpr size_t input_edge_size_{50};
    static constexpr size_t filter_edge_size_{27};
    static constexpr size_t top_padding_{13};
    static constexpr size_t left_padding_{13};
    static constexpr size_t right_padding_{left_padding_ + 1};
    static constexpr size_t bottom_padding_{top_padding_ + 1};
    static constexpr size_t padded_patch_edge_size_{input_edge_size_ + filter_edge_size_};
};

TEST(PatchAssembly, returnsCorrectPatchDimensionBasedOnFilterBorder) {
    const std::vector<size_t> filterBorders{27, 9, 7, 5};
    constexpr size_t patch_edge_size{50};
    const std::vector<size_t> expectedDimensions{
        patch_edge_size + filterBorders.at(0), patch_edge_size + filterBorders.at(1),
        patch_edge_size + filterBorders.at(2), patch_edge_size + filterBorders.at(3)};

    for (size_t i{0}; i < filterBorders.size(); i++) {
        ASSERT_THAT(alus::featurextractiongabor::GetPaddedPatchEdgeSize(patch_edge_size, filterBorders.at(i)),
                    Eq(expectedDimensions.at(i)));
    }
}

TEST(PatchAssembly, returnsCorrectPatchCountForArea) {
    constexpr size_t patch_side_size{50};
    ASSERT_THAT(alus::featurextractiongabor::GetPatchCountFor(1150, 892, patch_side_size), Eq(23 * 17));
    ASSERT_THAT(alus::featurextractiongabor::GetPatchCountFor(1199, 899, patch_side_size), Eq(23 * 17));
}

TEST(PatchAssembly, returnsCorrectPatchCountPerDimension) {
    constexpr size_t patch_side_size{50};
    ASSERT_THAT(alus::featurextractiongabor::GetPatchCountFor(1150, patch_side_size), Eq(23));
    ASSERT_THAT(alus::featurextractiongabor::GetPatchCountFor(892, patch_side_size), Eq(17));
}

TEST(PatchAssembly, returnsCorrectPaddedPatchImageDimensions) {
    const alus::RasterDimension input_dim{1150, 892};
    const std::vector<size_t> filter_edge_sizes{27, 9, 7, 5};
    constexpr size_t patch_side_size{50};
    const std::vector<alus::RasterDimension> expected_patch_image_dim{
        {1771, 1309}, {1357, 1003}, {1311, 969}, {1265, 935}};

    for (size_t i = 0; i < expected_patch_image_dim.size(); i++) {
        const auto res = alus::featurextractiongabor::GetPaddedPatchImageDimension(patch_side_size, input_dim,
                                                                                   filter_edge_sizes.at(i));

        ASSERT_THAT(res.columnsX, Eq(expected_patch_image_dim.at(i).columnsX));
        ASSERT_THAT(res.rowsY, Eq(expected_patch_image_dim.at(i).rowsY));
    }
}

TEST(PatchAssembly, createsPaddedPatchParametersCorrectly) {
    {
        const auto parameters = alus::featurextractiongabor::CreatePaddedPatchParameters(50, 27);
        ASSERT_THAT(parameters.padding_top, Eq(13));
        ASSERT_THAT(parameters.padding_left, Eq(13));
        ASSERT_THAT(parameters.padding_bottom, Eq(14));
        ASSERT_THAT(parameters.padding_right, Eq(14));
        ASSERT_THAT(parameters.origin_patch_edge_size, Eq(50));
        ASSERT_THAT(parameters.padded_patch_edge_size, Eq(50 + 27));
    }

    {
        const auto parameters = alus::featurextractiongabor::CreatePaddedPatchParameters(21, 10);
        ASSERT_THAT(parameters.padding_top, Eq(5));
        ASSERT_THAT(parameters.padding_left, Eq(5));
        ASSERT_THAT(parameters.padding_bottom, Eq(5));
        ASSERT_THAT(parameters.padding_right, Eq(5));
        ASSERT_THAT(parameters.origin_patch_edge_size, Eq(21));
        ASSERT_THAT(parameters.padded_patch_edge_size, Eq(21 + 10));
    }
}

TEST_F(PatchFullAssembly, copiesPatchToPaddedBufferCorrectly) {
    const auto& padded_patch_result = alus::featurextractiongabor::CreatePatchWithEmptyPadding(
        input_patch_data_,
        {top_padding_, left_padding_, bottom_padding_, right_padding_, input_edge_size_, padded_patch_edge_size_});
    ASSERT_THAT(padded_patch_edge_size_ * padded_patch_edge_size_, Eq(padded_patch_result.size()));
    ASSERT_THAT(padded_patch_result.size() * sizeof(float), Eq(expected_padded_patch_size_));

    size_t input_line_offset{0};
    for (size_t line = top_padding_; line < top_padding_ + input_edge_size_; line++) {
        const auto result_data_offset =
            line * padded_patch_edge_size_ + left_padding_;  // offset from top + offset from left
        for (size_t p = 0; p < input_edge_size_; p++) {
            ASSERT_THAT(static_cast<float>(input_patch_data_[input_line_offset * input_edge_size_ + p]),
                        FloatEq(padded_patch_result.at(result_data_offset + p)));
        }
        input_line_offset++;
    }
}

TEST_F(PatchFullAssembly, fillsLeftPaddingCorrectly) {
    const alus::featurextractiongabor::PaddedPatchParameters padding_parameters{
        top_padding_, left_padding_, bottom_padding_, right_padding_, input_edge_size_, padded_patch_edge_size_};
    auto padded_patch_result =
        alus::featurextractiongabor::CreatePatchWithEmptyPadding(input_patch_data_, padding_parameters);
    alus::featurextractiongabor::FillLeftPadding(padded_patch_result.data(), padding_parameters);
    for (size_t line = top_padding_; line < top_padding_ + input_edge_size_; line++) {
        const auto result_data_offset = line * padded_patch_edge_size_;
        for (size_t p = 0; p < left_padding_; p++) {
            ASSERT_THAT(padded_patch_result.at(result_data_offset + p),
                        FloatEq(expected_padded_patch_data_[result_data_offset + p]));
        }
    }
}

TEST_F(PatchFullAssembly, fillsRightPaddingCorrectly) {
    const alus::featurextractiongabor::PaddedPatchParameters padding_parameters{
        top_padding_, left_padding_, bottom_padding_, right_padding_, input_edge_size_, padded_patch_edge_size_};
    auto padded_patch_result =
        alus::featurextractiongabor::CreatePatchWithEmptyPadding(input_patch_data_, padding_parameters);
    alus::featurextractiongabor::FillRightPadding(padded_patch_result.data(), padding_parameters);
    for (size_t line = top_padding_; line < top_padding_ + input_edge_size_; line++) {
        const auto result_data_offset = line * padded_patch_edge_size_ + padding_parameters.padding_left +
                                        padding_parameters.origin_patch_edge_size;
        for (size_t p = 0; p < padding_parameters.padding_right; p++) {
            ASSERT_THAT(padded_patch_result.at(result_data_offset + p),
                        FloatEq(expected_padded_patch_data_[result_data_offset + p]));
        }
    }
}

TEST_F(PatchFullAssembly, fillsTopPaddingCorrectly) {
    const alus::featurextractiongabor::PaddedPatchParameters padding_parameters{
        top_padding_, left_padding_, bottom_padding_, right_padding_, input_edge_size_, padded_patch_edge_size_};
    auto padded_patch_result =
        alus::featurextractiongabor::CreatePatchWithEmptyPadding(input_patch_data_, padding_parameters);
    // Left and right paddings are needed to mirror data to top region
    alus::featurextractiongabor::FillLeftPadding(padded_patch_result.data(), padding_parameters);
    alus::featurextractiongabor::FillRightPadding(padded_patch_result.data(), padding_parameters);
    alus::featurextractiongabor::FillTopPadding(padded_patch_result.data(), padding_parameters);

    for (size_t line = 0; line < top_padding_; line++) {
        const auto result_data_offset = line * padded_patch_edge_size_;
        for (size_t p = 0; p < padding_parameters.padded_patch_edge_size; p++) {
            ASSERT_THAT(padded_patch_result.at(result_data_offset + p),
                        FloatEq(expected_padded_patch_data_[result_data_offset + p]));
        }
    }
}

TEST_F(PatchFullAssembly, fillsBottomPaddingCorrectly) {
    const alus::featurextractiongabor::PaddedPatchParameters padding_parameters{
        top_padding_, left_padding_, bottom_padding_, right_padding_, input_edge_size_, padded_patch_edge_size_};
    auto padded_patch_result =
        alus::featurextractiongabor::CreatePatchWithEmptyPadding(input_patch_data_, padding_parameters);
    // Left and right paddings are needed to mirror data to top region
    alus::featurextractiongabor::FillLeftPadding(padded_patch_result.data(), padding_parameters);
    alus::featurextractiongabor::FillRightPadding(padded_patch_result.data(), padding_parameters);
    alus::featurextractiongabor::FillBottomPadding(padded_patch_result.data(), padding_parameters);

    for (size_t line = top_padding_ + input_edge_size_; line < padded_patch_edge_size_; line++) {
        const auto result_data_offset = line * padded_patch_edge_size_;
        for (size_t p = 0; p < padding_parameters.padded_patch_edge_size; p++) {
            ASSERT_THAT(padded_patch_result.at(result_data_offset + p),
                        FloatEq(expected_padded_patch_data_[result_data_offset + p]));
        }
    }
}

TEST_F(PatchFullAssembly, fillPaddedPatch) {
    const alus::featurextractiongabor::PaddedPatchParameters padding_parameters{
        top_padding_, left_padding_, bottom_padding_, right_padding_, input_edge_size_, padded_patch_edge_size_};
    const auto& padded_patch_result =
        alus::featurextractiongabor::CreatePatchWithPaddingFilled(input_patch_data_, padding_parameters);

    ASSERT_THAT(padded_patch_result.size() * sizeof(float), Eq(expected_padded_patch_size_));

    for (size_t i = 0; i < padded_patch_result.size(); i++) {
        ASSERT_THAT(padded_patch_result.at(i), FloatEq(expected_padded_patch_data_[i]));
    }
}

}  // namespace