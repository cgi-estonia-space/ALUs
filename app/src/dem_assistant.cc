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
#include <boost/filesystem.hpp>

#include "gdal_util.h"
#include "zip_util.h"

namespace {
constexpr std::string_view TIF_EXTENSION{".tif"};
}

namespace alus::app {

bool DemAssistant::ArgumentsExtract::IsValid(std::string_view dem_file) {
    const auto extension = boost::filesystem::path(dem_file.data()).extension().string();
    bool is_valid = extension == TIF_EXTENSION;
    if (extension == gdal::constants::ZIP_EXTENSION) {
        const auto zip_contents = common::zip::GetZipContents(dem_file);
        is_valid |= std::any_of(std::begin(zip_contents), std::end(zip_contents), [](const auto& file) {
            return boost::filesystem::path(file).extension().string() == TIF_EXTENSION;
        });
    }
    return is_valid;
}

std::vector<std::string> DemAssistant::ArgumentsExtract::ExtractSrtm3Files(
    const std::vector<std::string>& cmd_line_arguments) {
    std::vector<std::string> srtm3_files{};
    for (auto&& arg : cmd_line_arguments) {
        std::vector<std::string> argument_values{};
        boost::split(argument_values, arg, boost::is_any_of("\t "));

        for (const auto& value : argument_values) {
            if (!IsValid(arg)) {
                throw std::invalid_argument("Invalid DEM file - " + arg);
            }

            srtm3_files.push_back(AdjustSrtm3Path(value));
        }
    }
    return srtm3_files;
}
std::string DemAssistant::ArgumentsExtract::AdjustSrtm3Path(std::string_view path) {
    if (const auto file_path = boost::filesystem::path(path.data());
        file_path.extension().string() == gdal::constants::ZIP_EXTENSION) {
        const auto tiff_name = boost::filesystem::change_extension(file_path.leaf(), TIF_EXTENSION.data());
        return gdal::constants::GDAL_ZIP_PREFIX.data() + file_path.string() + "/" + tiff_name.string();
    }
    return path.data();
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
