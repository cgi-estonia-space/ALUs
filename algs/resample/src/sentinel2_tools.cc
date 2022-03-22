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

#include "sentinel2_tools.h"

#include <algorithm>
#include <array>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

#include <boost/algorithm/string/classification.hpp>
#include <boost/algorithm/string/split.hpp>

#include "gdal_util.h"

namespace {
constexpr size_t SENTINEL2_FORMAT_FIELD_COUNT{7};
constexpr std::array<std::string_view, 2> SENTINEL2_MISSION_IDENTIFIERS{"S2A", "S2B"};
constexpr std::array<std::string_view, 2> SENTINEL2_SUPPORTED_LEVELS{"MSIL1C", "MSIL2A"};
}  // namespace

namespace alus::resample {

S2Fields TryParseS2Fields(std::string_view product) {
    std::string fields_only(product);
    if (const auto dot_pos = fields_only.find('.'); dot_pos != std::string::npos) {
        fields_only.erase(dot_pos, fields_only.length());
    }

    std::vector<std::string> fields_split;
    boost::split(fields_split, fields_only, boost::is_any_of("_"), boost::token_compress_on);

    if (fields_split.size() != SENTINEL2_FORMAT_FIELD_COUNT) {
        throw Sentinel2DatasetNameParseException(std::string(product).append(" is incorrect Sentinel 2 naming format"));
    }

    const auto mission = fields_split.at(0);
    if (std::find(SENTINEL2_MISSION_IDENTIFIERS.cbegin(), SENTINEL2_MISSION_IDENTIFIERS.cend(), mission) ==
        SENTINEL2_MISSION_IDENTIFIERS.cend()) {
        throw Sentinel2DatasetNameParseException(mission + " is not a valid mission");
    }

    const auto level = fields_split.at(1);
    if (std::find(SENTINEL2_SUPPORTED_LEVELS.cbegin(), SENTINEL2_SUPPORTED_LEVELS.cend(), level) ==
        SENTINEL2_SUPPORTED_LEVELS.cend()) {
        throw Sentinel2DatasetNameParseException(level + " is not a supported product level");
    }

    // NOLINTBEGIN
    return {mission,           level, fields_split.at(2), fields_split.at(3), fields_split.at(4), fields_split.at(5),
            fields_split.at(6)};
    // NOLINTEND
}

std::string TryCreateSentinel2GdalDatasetName(std::string_view path) {
    std::filesystem::path input(path);

    if (input.has_extension()) {
        const auto ext = input.extension().string();
        if (ext == ".xml") {
            TryParseS2Fields(input.parent_path().stem().c_str());
            return std::string(path);
        }

        if (ext == gdal::constants::ZIP_EXTENSION) {
            TryParseS2Fields(input.stem().c_str());
            return std::string(path);
        }

        if (ext == ".SAFE") {
            const auto& fields = TryParseS2Fields(input.filename().c_str());
            auto result_path = input;
            input.append(std::string("MTD_").append(fields.level).append(".xml"));
            return input.string();
        }

        throw Sentinel2DatasetNameParseException(std::string(path).append(" not a valid name for Sentinel 2 product."));
    }

    // If supplied folder name without '.SAFE' postfix.
    const auto& fields = TryParseS2Fields(input.stem().c_str());
    auto result_path = input;
    result_path.append(std::string("MTD_").append(fields.level).append(".xml"));
    return result_path.string();
}

std::unordered_map<std::string, std::string> ParseBandMetadata(char** metadata_list, size_t item_count) {
    std::unordered_map<std::string, std::string> md;
    for (size_t i = 0; i < item_count; i++) {
        std::vector<std::string> fields_split;
        boost::split(fields_split, metadata_list[i], boost::is_any_of("="), boost::token_compress_on);
        md.try_emplace(fields_split.front(), fields_split.back());
    }

    return md;
}
}  // namespace alus::resample