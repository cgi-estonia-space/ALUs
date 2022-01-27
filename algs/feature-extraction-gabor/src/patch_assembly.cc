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

#include <algorithm>
#include <stdexcept>

#include "../include/patch_assembly.h"

namespace alus::featurextractiongabor {

PaddedPatchParameters CreatePaddedPatchParameters(size_t origin_patch_edge_size, size_t filter_edge_size) {
    size_t asymmetrical_extra_pixel = filter_edge_size % 2;
    size_t filter_half_edge = filter_edge_size / 2;

    return {filter_half_edge,
            filter_half_edge,
            filter_half_edge + asymmetrical_extra_pixel,
            filter_half_edge + asymmetrical_extra_pixel,
            origin_patch_edge_size,
            origin_patch_edge_size + filter_edge_size};
}

alus::RasterDimension GetPaddedPatchImageDimension(size_t patch_edge_size,
                                                   const alus::RasterDimension& source_image_dim,
                                                   size_t filter_edge_size) {
    const auto ppes = GetPaddedPatchEdgeSize(patch_edge_size, filter_edge_size);
    const auto pcx = GetPatchCountFor(source_image_dim.columnsX, patch_edge_size);
    const auto pcy = GetPatchCountFor(source_image_dim.rowsY, patch_edge_size);

    return {static_cast<int>(pcx * ppes), static_cast<int>(pcy * ppes)};
}

std::vector<float> CreatePatchWithEmptyPadding(const uint8_t* input_patch, const PaddedPatchParameters& p) {

    std::vector<float> result = CreateEmptyPaddingPatch(p);
    FillCenterWithOrigin(input_patch, result.data(), p);

    return result;
}

void FillLeftPadding(float* patch_buffer, const PaddedPatchParameters& p) {
    for (size_t line = p.padding_top; line < p.padding_top + p.origin_patch_edge_size; line++) {
        const auto row_start_offset_index = line * p.padded_patch_edge_size;
        const auto padding_data_offset_index = row_start_offset_index + p.padding_left;
        std::reverse_copy(patch_buffer + padding_data_offset_index,
                          patch_buffer + padding_data_offset_index + p.padding_left,
                          patch_buffer + row_start_offset_index);
    }
}

void FillRightPadding(float* patch_buffer, const PaddedPatchParameters& p) {
    for (size_t line = p.padding_top; line < p.padding_top + p.origin_patch_edge_size; line++) {
        const auto padding_right_start_offset_index =
            line * p.padded_patch_edge_size + p.padding_left + p.origin_patch_edge_size;
        const auto padding_data_offset_index = padding_right_start_offset_index - p.padding_right;
        std::reverse_copy(patch_buffer + padding_data_offset_index,
                          patch_buffer + padding_data_offset_index + p.padding_right,
                          patch_buffer + padding_right_start_offset_index);
    }
}

void FillTopPadding(float* patch_buffer, const PaddedPatchParameters& p) {
    for (size_t line = 0; line < p.padding_top; line++) {
        const auto padding_source_index = (2 * p.padding_top - 1 - line) * p.padded_patch_edge_size;
        const auto padding_destination_index = line * p.padded_patch_edge_size;
        std::copy(patch_buffer + padding_source_index, patch_buffer + padding_source_index + p.padded_patch_edge_size,
                  patch_buffer + padding_destination_index);
    }
}

void FillBottomPadding(float* patch_buffer, const PaddedPatchParameters& p) {
    const auto bottom_padding_start_line = p.padded_patch_edge_size - p.padding_bottom;
    for (size_t line = bottom_padding_start_line; line < p.padded_patch_edge_size; line++) {
        const auto padding_source_line = (bottom_padding_start_line - (line - bottom_padding_start_line)) - 1;
        const auto padding_source_index = padding_source_line * p.padded_patch_edge_size;
        const auto padding_destination_index = line * p.padded_patch_edge_size;
        std::copy(patch_buffer + padding_source_index, patch_buffer + padding_source_index + p.padded_patch_edge_size,
                  patch_buffer + padding_destination_index);
    }
}
std::vector<float> CreatePatchWithPaddingFilled(const uint8_t* input_patch, const PaddedPatchParameters& parameters) {
    auto patch = CreatePatchWithEmptyPadding(input_patch, parameters);
    FillLeftPadding(patch.data(), parameters);
    FillRightPadding(patch.data(), parameters);
    FillTopPadding(patch.data(), parameters);
    FillBottomPadding(patch.data(), parameters);

    return patch;
}

}  // namespace alus::featurextractiongabor