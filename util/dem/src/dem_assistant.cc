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

#include <algorithm>
#include <filesystem>
#include <stdexcept>
#include <string_view>

#include <boost/algorithm/string.hpp>

#include "copdem_cog_30m.h"
#include "gdal_util.h"
#include "srtm3_elevation_model.h"
#include "zip_util.h"

namespace {
constexpr std::string_view TIF_EXTENSION{".tif"};
}

namespace alus::dem {

bool Assistant::ArgumentsExtract::IsValidSrtm3Filename(std::string_view dem_file) {
    const auto extension = std::filesystem::path(dem_file.data()).extension().string();
    bool is_valid = extension == TIF_EXTENSION;
    if (extension == gdal::constants::ZIP_EXTENSION) {
        const auto zip_contents = common::zip::GetZipContents(dem_file);
        is_valid |= std::any_of(std::begin(zip_contents), std::end(zip_contents), [](const auto& file) {
            return std::filesystem::path(file).extension().string() == TIF_EXTENSION;
        });
    }

    if (!is_valid) {
        return is_valid;
    }

    std::vector<std::string> filename_items{};
    boost::split(filename_items, std::filesystem::path(dem_file).stem().string(), boost::is_any_of("_"));
    // Example - 'srtm_41_40.tif'
    if (filename_items.size() != 3) {
        return false;
    }

    constexpr std::string_view begin_pattern{"srtm"};
    return filename_items.front() == begin_pattern;
}

std::vector<std::string> Assistant::ArgumentsExtract::ExtractSrtm3Files(
    const std::vector<std::string>& cmd_line_arguments) {
    std::vector<std::string> srtm3_files{};
    for (auto&& arg : cmd_line_arguments) {
        std::vector<std::string> argument_values{};
        boost::split(argument_values, arg, boost::is_any_of("\t "));

        for (const auto& value : argument_values) {
            if (!IsValidSrtm3Filename(value)) {
                throw std::invalid_argument("Following filename - '" + value + "' - is not a SRTM3 format");
            }

            srtm3_files.push_back(AdjustZipPathForSrtm3(value));
        }
    }
    return srtm3_files;
}

bool Assistant::ArgumentsExtract::IsValidCopDemCog30mFilename(std::string_view filename) {
    const auto extension = std::filesystem::path(filename).extension().string();
    if (extension != TIF_EXTENSION) {
        return false;
    }

    const auto filename_stem = std::filesystem::path(filename).stem().string();
    std::vector<std::string> filename_items{};
    boost::split(filename_items, filename_stem, boost::is_any_of("_"));
    // Example - 'Copernicus_DSM_COG_10_S18_00_E020_00_DEM.tif'
    if (filename_items.size() != 9) {
        return false;
    }

    constexpr std::string_view begin_pattern{"Copernicus_DSM_COG_10"};
    return filename_stem.substr(0, begin_pattern.length()) == begin_pattern;
}

std::string Assistant::ArgumentsExtract::AdjustZipPathForSrtm3(std::string_view path) {
    if (const auto file_path = std::filesystem::path(path.data());
        file_path.extension().string() == gdal::constants::ZIP_EXTENSION) {
        const auto tiff_name = file_path.filename().stem().string() + TIF_EXTENSION.data();
        return gdal::constants::GDAL_ZIP_PREFIX.data() + file_path.string() + "/" + tiff_name;
    }
    return path.data();
}

std::vector<std::string> Assistant::ArgumentsExtract::PrepareArgs(const std::vector<std::string>& cmd_line_arguments) {
    std::vector<std::string> dem_files{};
    for (auto&& arg : cmd_line_arguments) {
        std::vector<std::string> argument_values{};
        boost::split(argument_values, arg, boost::is_any_of("\t "));

        for (const auto& value : argument_values) {
            if (value.empty()) {
                continue;
            }
            dem_files.push_back(value);
        }
    }

    return dem_files;
}

std::shared_ptr<Assistant> Assistant::CreateFormattedDemTilesOnGpuFrom(
    const std::vector<std::string>& cmd_line_arguments) {
    const auto filenames = ArgumentsExtract::PrepareArgs(cmd_line_arguments);
    if (std::all_of(filenames.cbegin(), filenames.cend(), Assistant::ArgumentsExtract::IsValidCopDemCog30mFilename)) {
        return std::make_shared<Assistant>(std::move(filenames), Type::COPDEM_COG30m);
    } else if (std::all_of(filenames.cbegin(), filenames.cend(), Assistant::ArgumentsExtract::IsValidSrtm3Filename)) {
        return std::make_shared<Assistant>(Assistant::ArgumentsExtract::ExtractSrtm3Files(std::move(filenames)),
                                           Type::SRTM3);
    } else {
        throw std::invalid_argument("Not supported DEM filenames detected.");
    }
}

Assistant::Assistant(std::vector<std::string> filenames, Type mission)
    : type_{mission}, egm96_{std::make_shared<snapengine::EarthGravitationalModel96>()} {
    egm96_->HostToDevice();
    switch (mission) {
        case Type::SRTM3:
            model_ = std::make_shared<snapengine::Srtm3ElevationModel>(std::move(filenames), egm96_);
            break;
        case Type::COPDEM_COG30m:
            model_ = std::make_shared<CopDemCog30m>(std::move(filenames), egm96_);
            break;
        default:
            throw std::runtime_error("This code should not reach to unknown DEM type.");
    }
    model_->LoadTiles();
}

}  // namespace alus::dem
