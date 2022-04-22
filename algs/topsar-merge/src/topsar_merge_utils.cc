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
#include "topsar_merge_utils.h"

#include <cmath>
#include <cstddef>
#include <map>
#include <string>
#include <string_view>

#include "s1tbx-commons/subswath_info.h"
#include "shapes.h"
#include "snap-engine-utilities/engine-utilities/eo/constants.h"

namespace alus::topsarmerge {
namespace {
constexpr double LINE_INDEX_SHIFT{0.5};
constexpr double SAMPLE_INDEX_SHIFT{0.5};
}  // namespace

void FindFirstAndLastSubSwathIndices(int& first_index, int& last_index, const Rectangle& target_rectangle,
                                     const MergeOperatorParameters& operator_parameters,
                                     const std::map<sub_swath_map_index, SubSwathMergeInfo>& sub_swath_merge_info_map) {
    const auto tile_slrt_to_first_pixel = operator_parameters.target_slant_range_time_to_first_pixel +
                                          target_rectangle.x * operator_parameters.target_delta_slant_range_time;
    const auto tile_slrt_to_last_pixel =
        operator_parameters.target_slant_range_time_to_first_pixel +
        (target_rectangle.x + target_rectangle.width - 1) * operator_parameters.target_delta_slant_range_time;
    const auto tile_first_line_time =
        operator_parameters.target_first_line_time + target_rectangle.y * operator_parameters.target_line_time_interval;
    const auto tile_last_line_time =
        operator_parameters.target_first_line_time +
        (target_rectangle.y + target_rectangle.height - 1) * operator_parameters.target_line_time_interval;

    auto is_valid_for_first_sub_swath = [&tile_slrt_to_first_pixel, &tile_first_line_time,
                                         &tile_last_line_time](const SubSwathMergeInfo& sub_swath_info) {
        return (tile_slrt_to_first_pixel >= sub_swath_info.slr_time_to_first_valid_pixel &&
                tile_slrt_to_first_pixel <= sub_swath_info.slr_time_to_last_valid_pixel) &&
               ((tile_first_line_time >= sub_swath_info.first_valid_line_time &&
                 tile_first_line_time < sub_swath_info.last_valid_line_time) ||
                (tile_last_line_time >= sub_swath_info.first_valid_line_time &&
                 tile_last_line_time < sub_swath_info.last_valid_line_time));
    };

    auto is_valid_for_last_sub_swath = [&tile_slrt_to_last_pixel, &tile_first_line_time,
                                        &tile_last_line_time](const SubSwathMergeInfo& sub_swath_info) {
        return (tile_slrt_to_last_pixel >= sub_swath_info.slr_time_to_first_valid_pixel &&
                tile_slrt_to_last_pixel <= sub_swath_info.slr_time_to_last_valid_pixel) &&
               ((tile_first_line_time >= sub_swath_info.first_valid_line_time &&
                 tile_first_line_time < sub_swath_info.last_valid_line_time) ||
                (tile_last_line_time >= sub_swath_info.first_valid_line_time &&
                 tile_last_line_time < sub_swath_info.last_valid_line_time));
    };

    for (size_t i = 0; i < operator_parameters.number_of_subswaths; i++) {
        const auto& sub_swath_merge_info = sub_swath_merge_info_map.at(i);
        if (is_valid_for_first_sub_swath(sub_swath_merge_info)) {
            first_index = static_cast<int>(i);
            break;
        }
    }

    for (size_t i = 0; i < operator_parameters.number_of_subswaths; i++) {
        const auto& sub_swath_merge_info = sub_swath_merge_info_map.at(i);
        if (is_valid_for_last_sub_swath(sub_swath_merge_info)) {
            last_index = static_cast<int>(i);
        }
    }

    if (first_index != -1 && last_index == -1) {
        last_index = first_index;
    }

    if (first_index == -1 && last_index != -1) {
        first_index = last_index;
    }
}
void ComputeTargetStartEndTime(MergeOperatorParameters& operator_parameters,
                               const std::map<sub_swath_map_index, SubSwathMergeInfo>& merge_info_map) {
    operator_parameters.target_first_line_time = merge_info_map.at(0).first_line_time;
    operator_parameters.target_last_line_time = merge_info_map.at(0).last_line_time;
    for (size_t i = 0; i < operator_parameters.number_of_subswaths; i++) {
        if (auto first_time = merge_info_map.at(i).first_line_time;
            operator_parameters.target_first_line_time > first_time) {
            operator_parameters.target_first_line_time = first_time;
        }
        if (auto last_time = merge_info_map.at(i).last_line_time;
            operator_parameters.target_last_line_time < last_time) {
            operator_parameters.target_last_line_time = last_time;
        }
    }
    operator_parameters.target_line_time_interval = merge_info_map.at(0).azimuth_time_interval;
}
int GetSubSwathIndexFromName(std::string_view sub_swath_name) { return std::stoi(sub_swath_name.substr(2).data()); }
int GetSubSwathIndex(int target_x, int target_y, int first_sub_swath_index, int last_sub_swath_index,
                     const MergeOperatorParameters& operator_parameters,
                     const std::map<sub_swath_map_index, SubSwathMergeInfo>& sub_swath_merge_info_map) {
    const auto target_sample_slr_time = operator_parameters.target_slant_range_time_to_first_pixel +
                                        target_x * operator_parameters.target_delta_slant_range_time;
    const auto target_line_time =
        operator_parameters.target_first_line_time + target_y * operator_parameters.target_line_time_interval;

    int count{0};
    int swath_0{-1};
    int swath_1{-1};
    for (int i = first_sub_swath_index; i <= last_sub_swath_index; ++i) {
        const auto& info = sub_swath_merge_info_map.at(i);
        if (target_line_time >= info.first_valid_line_time && target_line_time <= info.last_valid_line_time &&
            target_sample_slr_time >= info.slr_time_to_first_valid_pixel &&
            target_sample_slr_time <= info.slr_time_to_last_valid_pixel) {
            if (count == 0) {
                swath_0 = i;
            } else {
                swath_1 = i;
                break;
            }
            ++count;
        }
    }

    if (swath_1 != -1) {
        const auto middle_time = (sub_swath_merge_info_map.at(swath_0).slr_time_to_last_valid_pixel +
                                  sub_swath_merge_info_map.at(swath_1).slr_time_to_first_valid_pixel) /
                                 2.0;
        if (target_sample_slr_time > middle_time) {
            return swath_1;
        }
    }
    return swath_0;
}
size_t GetSubSwathIndexBySlrTime(double slant_range_time, const MergeOperatorParameters& operator_parameters,
                                 const std::map<sub_swath_map_index, SubSwathMergeInfo>& sub_swath_merge_info_map) {
    for (size_t i = 0; i < operator_parameters.number_of_subswaths; i++) {
        double start_time{};
        double end_time{};

        if (i == 0) {
            start_time = sub_swath_merge_info_map.at(i).slr_time_to_first_valid_pixel;
        } else {
            start_time = 0.5 * (sub_swath_merge_info_map.at(i).slr_time_to_first_valid_pixel +
                                sub_swath_merge_info_map.at(i - 1).slr_time_to_last_pixel);
        }

        if (i == operator_parameters.number_of_subswaths - 1) {
            end_time = sub_swath_merge_info_map.at(i).slr_time_to_last_pixel;
        } else {
            end_time = 0.5 * (sub_swath_merge_info_map.at(i).slr_time_to_last_pixel +
                              sub_swath_merge_info_map.at(i + 1).slr_time_to_first_valid_pixel);
        }

        if (slant_range_time >= start_time && slant_range_time < end_time) {
            return i;
        }
    }

    return 0;
}
std::string GetTargetBandNameFromSourceBandName(std::string_view source_band_name, std::string_view acquisition_mode) {
    if (source_band_name.find(acquisition_mode) == std::string::npos) {
        return std::string(source_band_name);
    }

    const auto first_separation_position = source_band_name.find(acquisition_mode);
    const auto second_separation_position = source_band_name.find('_', first_separation_position + 1);
    const auto t1 = source_band_name.substr(0, first_separation_position);
    const auto t2 = source_band_name.substr(second_separation_position + 1);
    return std::string(t1) + t2.data();
}

int GetSampleIndexInSourceProduct(int target_x, int number_of_samples, double slr_time_to_first_pixel,
                                  const MergeOperatorParameters& operator_parameters) {
    const auto source_x =
        static_cast<int>((((operator_parameters.target_slant_range_time_to_first_pixel +
                            target_x * operator_parameters.target_delta_slant_range_time - slr_time_to_first_pixel) /
                           operator_parameters.target_delta_slant_range_time) +
                          SAMPLE_INDEX_SHIFT));

    if (source_x < 0) {
        return 0;
    }
    return source_x > number_of_samples - 1 ? number_of_samples - 1 : source_x;
}
int GetLineIndexInSourceProduct(int target_y, const SubSwathMergeInfo& sub_swath_merge_info,
                                const MergeOperatorParameters& operator_parameters) {
    const auto target_line_time =
        operator_parameters.target_first_line_time + target_y * operator_parameters.target_line_time_interval;
    const auto source_y = static_cast<int>((target_line_time - sub_swath_merge_info.first_line_time) /
                                               sub_swath_merge_info.azimuth_time_interval +
                                           LINE_INDEX_SHIFT);

    if (source_y < 0) {
        return 0;
    }
    return source_y > sub_swath_merge_info.number_of_lines - 1 ? sub_swath_merge_info.number_of_lines - 1 : source_y;
}
int ComputeYMin(const SubSwathMergeInfo& sub_swath_merge_info, const MergeOperatorParameters& operator_parameters) {
    return static_cast<int>(
        std::round((sub_swath_merge_info.first_line_time - operator_parameters.target_first_line_time) /
                   operator_parameters.target_line_time_interval));
}
int ComputeYMax(const SubSwathMergeInfo& sub_swath_merge_info, const MergeOperatorParameters& operator_parameters) {
    return static_cast<int>(
        std::round((sub_swath_merge_info.last_line_time - operator_parameters.target_first_line_time) /
                   operator_parameters.target_line_time_interval));
}
int ComputeXMin(const SubSwathMergeInfo& sub_swath_merge_info, const MergeOperatorParameters& operator_parameters) {
    return static_cast<int>(std::round((sub_swath_merge_info.slr_time_to_first_valid_pixel -
                                        operator_parameters.target_slant_range_time_to_first_pixel) /
                                       operator_parameters.target_delta_slant_range_time));
}
int ComputeXMax(const SubSwathMergeInfo& sub_swath_merge_info, const MergeOperatorParameters& operator_parameters) {
    return static_cast<int>(std::round((sub_swath_merge_info.slr_time_to_last_valid_pixel -
                                        operator_parameters.target_slant_range_time_to_first_pixel) /
                                       operator_parameters.target_delta_slant_range_time));
}
std::vector<Rectangle> GetSourceRectangles(
    int number_of_source_tiles, int first_sub_swath_index, int last_sub_swath_index, const Rectangle& target_rectangle,
    const MergeOperatorParameters& operator_parameters,
    const std::map<sub_swath_map_index, SubSwathMergeInfo>& sub_swath_merge_info_map) {
    std::vector<Rectangle> source_rectangles;
    source_rectangles.reserve(number_of_source_tiles);

    for (int i = first_sub_swath_index; i <= last_sub_swath_index; ++i) {
        const auto& sub_swath_merge_info = sub_swath_merge_info_map.at(i);

        const auto x_0 =
            GetSampleIndexInSourceProduct(target_rectangle.x, sub_swath_merge_info.number_of_samples,
                                          sub_swath_merge_info.slr_time_to_first_pixel, operator_parameters);
        const auto x_max = GetSampleIndexInSourceProduct(
            target_rectangle.x + target_rectangle.width - 1, sub_swath_merge_info.number_of_samples,
            sub_swath_merge_info.slr_time_to_first_pixel, operator_parameters);

        const auto y_0 = GetLineIndexInSourceProduct(target_rectangle.y, sub_swath_merge_info, operator_parameters);
        const auto y_max = GetLineIndexInSourceProduct(target_rectangle.y + target_rectangle.height - 1,
                                                       sub_swath_merge_info, operator_parameters);

        source_rectangles.push_back({x_0, y_0, x_max - x_0 + 1, y_max - y_0 + 1});
    }

    return source_rectangles;
}
void ComputeTargetSlantRangeTimeToFirstAndLastPixels(
    MergeOperatorParameters& operator_parameters,
    const std::map<sub_swath_map_index, SubSwathMergeInfo>& sub_swath_merge_info_map) {
    operator_parameters.target_slant_range_time_to_first_pixel =
        sub_swath_merge_info_map.at(0).slr_time_to_first_valid_pixel;
    operator_parameters.target_slant_range_time_to_last_pixel =
        sub_swath_merge_info_map.at(operator_parameters.number_of_subswaths - 1).slr_time_to_last_valid_pixel;
    operator_parameters.target_delta_slant_range_time =
        sub_swath_merge_info_map.at(0).range_pixel_spacing / snapengine::eo::constants::LIGHT_SPEED;
}
void ComputeTargetWidthAndHeight(MergeOperatorParameters& operator_parameters) {
    operator_parameters.target_height =
        static_cast<int>((operator_parameters.target_last_line_time - operator_parameters.target_first_line_time) /
                         operator_parameters.target_line_time_interval);
    operator_parameters.target_width = static_cast<int>((operator_parameters.target_slant_range_time_to_last_pixel -
                                                         operator_parameters.target_slant_range_time_to_first_pixel) /
                                                        operator_parameters.target_delta_slant_range_time);
}
SubSwathMergeInfo GetSubSwathMergeInfoFromSentinel1SubSwathInfo(const s1tbx::SubSwathInfo& info) {
    return {info.slr_time_to_first_valid_pixel_,
            info.slr_time_to_last_valid_pixel_,
            info.first_valid_line_time_,
            info.last_valid_line_time_,
            info.first_line_time_,
            info.last_line_time_,
            info.azimuth_time_interval_,
            info.slr_time_to_first_pixel_,
            info.slr_time_to_last_pixel_,
            info.num_of_samples_,
            info.num_of_lines_,
            info.range_pixel_spacing_};
}
}  // namespace alus::topsarmerge
