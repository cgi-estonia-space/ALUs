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

#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>

namespace alus::resample {

class Sentinel2DatasetNameParseException final : public std::runtime_error {
    using std::runtime_error::runtime_error;
};

struct S2Fields {
    std::string mission;
    std::string level;
    std::string sensing_start;
    std::string processing_baseline;
    std::string orbit;
    std::string tile;
    std::string discriminator;
};

S2Fields TryParseS2Fields(std::string_view product);
std::string TryCreateSentinel2GdalDatasetName(std::string_view path);
std::unordered_map<std::string, std::string> ParseBandMetadata(char** metadata_list, size_t item_count);

}  // namespace alus::resample