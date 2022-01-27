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
#pragma once

#include <cstddef>
#include <vector>

#include "raster_properties.h"

namespace alus::featurextractiongabor {

inline size_t GetPaddedPatchEdgeSize(size_t patch_edge_size, size_t filter_border_edge_size) {
    return patch_edge_size + filter_border_edge_size;
}

inline size_t GetPatchCountFor(size_t dim, size_t patch_side_size) { return dim / patch_side_size; }

inline size_t GetPatchCountFor(size_t dim1, size_t dim2, size_t patch_side_size) {
    return GetPatchCountFor(dim1, patch_side_size) * GetPatchCountFor(dim2, patch_side_size);
}

struct PaddedPatchParameters {
    size_t padding_top;
    size_t padding_left;
    size_t padding_bottom;
    size_t padding_right;
    size_t origin_patch_edge_size;
    size_t padded_patch_edge_size;
};
PaddedPatchParameters CreatePaddedPatchParameters(size_t origin_patch_edge_size, size_t filter_edge_size);

alus::RasterDimension GetPaddedPatchImageDimension(size_t patch_edge_size,
                                                   const alus::RasterDimension& source_image_dim,
                                                   size_t filter_edge_size);

inline std::vector<float> CreateEmptyPaddingPatch(const PaddedPatchParameters& parameters) {
    return std::vector<float>(parameters.padded_patch_edge_size * parameters.padded_patch_edge_size);
}

std::vector<float> CreatePatchWithEmptyPadding(const uint8_t* input_patch, const PaddedPatchParameters& parameters);

template <typename T>
void FillCenterWithOrigin(const T* origin, float* patch_with_padding, const PaddedPatchParameters& parameters) {
    for (size_t line = parameters.padding_top; line < parameters.origin_patch_edge_size + parameters.padding_top;
         line++) {
        for (size_t pi = 0; pi < parameters.origin_patch_edge_size; pi++) {
            patch_with_padding[line * parameters.padded_patch_edge_size + parameters.padding_left + pi] =
                static_cast<float>(origin[(line - parameters.padding_top) * parameters.origin_patch_edge_size + pi]);
        }
    }
}

struct StrideFillParameters {
    size_t origin_line_stride;
    size_t destination_line_stride;
};

template <typename T>
void FillCenterWithOrigin(const T* origin, float* patch_with_padding, const PaddedPatchParameters& parameters,
                          const StrideFillParameters& stride_parameters) {
    for (size_t line = parameters.padding_top; line < parameters.origin_patch_edge_size + parameters.padding_top;
         line++) {
        for (size_t pi = 0; pi < parameters.origin_patch_edge_size; pi++) {
            const size_t from_offset{(line - parameters.padding_top) * stride_parameters.origin_line_stride + pi};
            const size_t out_offset{line * stride_parameters.destination_line_stride + parameters.padding_left + pi};
            patch_with_padding[out_offset] = static_cast<float>(origin[from_offset]);
        }
    }
}

void FillLeftPadding(float* patch_buffer, const PaddedPatchParameters& parameters);
inline void FillLeftPadding(float* patch_buffer, const PaddedPatchParameters& parameters,
                            const StrideFillParameters& stride_parameters) {
    for (size_t line = parameters.padding_top; line < parameters.padding_top + parameters.origin_patch_edge_size;
         line++) {
        const auto padding_data_offset_index =
            line * stride_parameters.destination_line_stride + parameters.padding_left;
        std::reverse_copy(patch_buffer + padding_data_offset_index,
                          patch_buffer + padding_data_offset_index + parameters.padding_left,
                          patch_buffer + padding_data_offset_index - parameters.padding_left);
    }
}
void FillRightPadding(float* patch_buffer, const PaddedPatchParameters& parameters);
inline void FillRightPadding(float* patch_buffer, const PaddedPatchParameters& parameters,
                             const StrideFillParameters& stride_parameters) {
    for (size_t line = parameters.padding_top; line < parameters.padding_top + parameters.origin_patch_edge_size;
         line++) {
        const auto padding_data_offset_index = line * stride_parameters.destination_line_stride +
                                               parameters.padding_left + parameters.origin_patch_edge_size;
        std::reverse_copy(patch_buffer + padding_data_offset_index - parameters.padding_right,
                          patch_buffer + padding_data_offset_index, patch_buffer + padding_data_offset_index);
    }
}

void FillTopPadding(float* patch_buffer, const PaddedPatchParameters& parameters);
inline void FillTopPadding(float* patch_buffer, const PaddedPatchParameters& parameters,
                           const StrideFillParameters& stride_parameters) {
    for (size_t line = 0; line < parameters.padding_top; line++) {
        const auto padding_source_index =
            (2 * parameters.padding_top - 1 - line) * stride_parameters.destination_line_stride;
        const auto padding_destination_index = line * stride_parameters.destination_line_stride;
        std::copy(patch_buffer + padding_source_index,
                  patch_buffer + padding_source_index + parameters.padded_patch_edge_size,
                  patch_buffer + padding_destination_index);
    }
}

void FillBottomPadding(float* patch_buffer, const PaddedPatchParameters& parameters);
inline void FillBottomPadding(float* patch_buffer, const PaddedPatchParameters& parameters,
                              const StrideFillParameters& stride_parameters) {
    const auto bottom_padding_start_line = parameters.padded_patch_edge_size - parameters.padding_bottom;
    for (size_t line = bottom_padding_start_line; line < parameters.padded_patch_edge_size; line++) {
        const auto padding_source_line = (bottom_padding_start_line - (line - bottom_padding_start_line)) - 1;
        const auto padding_source_index = padding_source_line * stride_parameters.destination_line_stride;
        const auto padding_destination_index = line * stride_parameters.destination_line_stride;
        std::copy(patch_buffer + padding_source_index,
                  patch_buffer + padding_source_index + parameters.padded_patch_edge_size,
                  patch_buffer + padding_destination_index);
    }
}

std::vector<float> CreatePatchWithPaddingFilled(const uint8_t* input_patch, const PaddedPatchParameters& parameters);
template <typename T>
void FillPaddingPatch(const T* origin, float* patch, const PaddedPatchParameters& parameters,
                      const StrideFillParameters& stride_parameters) {
    FillCenterWithOrigin(origin, patch, parameters, stride_parameters);
    FillLeftPadding(patch, parameters, stride_parameters);
    FillRightPadding(patch, parameters, stride_parameters);
    FillTopPadding(patch, parameters, stride_parameters);
    FillBottomPadding(patch, parameters, stride_parameters);
}

}  // namespace alus::featurextractiongabor