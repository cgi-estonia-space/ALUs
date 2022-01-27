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
#include <map>
#include <string>
#include <string_view>

#include "s1tbx-commons/subswath_info.h"
#include "shapes.h"

namespace alus::topsarmerge {
struct MergeOperatorParameters {
    size_t number_of_subswaths{0};
    std::string acquisition_mode;
    std::string product_type;
    size_t reference_sub_swath_index{0};
    double target_first_line_time{};
    double target_last_line_time{};
    double target_line_time_interval{};
    double target_slant_range_time_to_first_pixel{};
    double target_slant_range_time_to_last_pixel{};
    double target_delta_slant_range_time{};
    int target_height{};
    int target_width{};
};

using sub_swath_map_index = size_t;  // Increases readability of map types
using product_map_index = size_t;    // Increases readability of map types

struct SubSwathMergeInfo {
    double slr_time_to_first_valid_pixel;
    double slr_time_to_last_valid_pixel;
    double first_valid_line_time;
    double last_valid_line_time;
    double first_line_time;
    double last_line_time;
    double azimuth_time_interval;
    double slr_time_to_first_pixel;
    double slr_time_to_last_pixel;
    int number_of_samples;
    int number_of_lines;
    double range_pixel_spacing;
};

SubSwathMergeInfo GetSubSwathMergeInfoFromSentinel1SubSwathInfo(const s1tbx::SubSwathInfo& info);

void FindFirstAndLastSubSwathIndices(int& first_index, int& last_index, const Rectangle& target_rectangle,
                                     const MergeOperatorParameters& operator_parameters,
                                     const std::map<sub_swath_map_index, SubSwathMergeInfo>& sub_swath_merge_info_map);
void ComputeTargetStartEndTime(MergeOperatorParameters& operator_parameters,
                               const std::map<sub_swath_map_index, SubSwathMergeInfo>& merge_info_map);
void ComputeTargetSlantRangeTimeToFirstAndLastPixels(
    MergeOperatorParameters& operator_parameters,
    const std::map<sub_swath_map_index, SubSwathMergeInfo>& sub_swath_merge_info_map);
void ComputeTargetWidthAndHeight(MergeOperatorParameters& operator_parameters);

[[nodiscard]] int GetSubSwathIndexFromName(std::string_view sub_swath_name);
[[nodiscard]] int GetSubSwathIndex(int target_x, int target_y, int first_sub_swath_index, int last_sub_swath_index,
                                   const MergeOperatorParameters& operator_parameters,
                                   const std::map<sub_swath_map_index, SubSwathMergeInfo>& sub_swath_merge_info_map);
[[nodiscard]] size_t GetSubSwathIndexBySlrTime(
    double slant_range_time, const MergeOperatorParameters& operator_parameters,
    const std::map<sub_swath_map_index, SubSwathMergeInfo>& sub_swath_merge_info_map);
[[nodiscard]] std::string GetTargetBandNameFromSourceBandName(std::string_view source_band_name,
                                                              std::string_view acquisition_mode);
[[nodiscard]] int GetSampleIndexInSourceProduct(int target_x, int number_of_samples, double slr_time_to_first_pixel,
                                                const MergeOperatorParameters& operator_parameters);
[[nodiscard]] int GetLineIndexInSourceProduct(int target_y, const SubSwathMergeInfo& sub_swath_merge_info,
                                              const MergeOperatorParameters& operator_parameters);
[[nodiscard]] int ComputeYMin(const SubSwathMergeInfo& sub_swath_merge_info,
                              const MergeOperatorParameters& operator_parameters);
[[nodiscard]] int ComputeYMax(const SubSwathMergeInfo& sub_swath_merge_info,
                              const MergeOperatorParameters& operator_parameters);
[[nodiscard]] int ComputeXMin(const SubSwathMergeInfo& sub_swath_merge_info,
                              const MergeOperatorParameters& operator_parameters);
[[nodiscard]] int ComputeXMax(const SubSwathMergeInfo& sub_swath_merge_info,
                              const MergeOperatorParameters& operator_parameters);
[[nodiscard]] std::vector<Rectangle> GetSourceRectangles(
    int number_of_source_tiles, int first_sub_swath_index, int last_sub_swath_index, const Rectangle& target_rectangle,
    const MergeOperatorParameters& operator_parameters,
    const std::map<sub_swath_map_index, SubSwathMergeInfo>& sub_swath_merge_info_map);
}  // namespace alus::topsarmerge