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

#include "dem_assistant.h"

#include <stdexcept>
#include <string_view>

#include <boost/algorithm/string.hpp>

namespace {
constexpr std::string_view TIF_EXTENSION{".tif"};
}

namespace alus::app {

bool DemAssistant::ArgumentsExtract::IsValid(std::string_view dem_file) {
    return dem_file.substr(dem_file.length() - TIF_EXTENSION.length(), dem_file.length()) == TIF_EXTENSION;
}

std::vector<std::string> DemAssistant::ArgumentsExtract::ExtractSrtm3Files(
    const std::vector<std::string>& cmd_line_arguments) {
    std::vector<std::string> srtm3_files{};
    for (auto&& arg : cmd_line_arguments) {
        std::vector<std::string> argument_values{};
        boost::split(argument_values, arg, boost::is_any_of("\t "));

        for (auto&& value : argument_values) {
            if (!IsValid(arg)) {
                throw std::invalid_argument("Invalid DEM file - " + arg);
            }
            srtm3_files.push_back(value);
        }
    }
    return srtm3_files;
}

std::shared_ptr<DemAssistant> DemAssistant::CreateFormattedSrtm3TilesOnGpuFrom(
    const std::vector<std::string>& cmd_line_arguments) {
    return std::make_shared<DemAssistant>(DemAssistant::ArgumentsExtract::ExtractSrtm3Files(cmd_line_arguments));
}

DemAssistant::DemAssistant(std::vector<std::string> srtm3_files)
    : model_(std::move(srtm3_files)), egm96_{std::make_shared<snapengine::EarthGravitationalModel96>()} {
    egm96_->HostToDevice();
    model_.ReadSrtmTiles(egm96_);
}

}  // namespace alus::app
