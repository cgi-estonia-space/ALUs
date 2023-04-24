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

#include <boost/algorithm/string/case_conv.hpp>

#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string_view>

#include "ceres-core/core/zip.h"
#include "dataset.h"
#include "gdal_util.h"
#include "zip_util.h"

namespace alus::dataset {

template <typename T>
std::shared_ptr<T> OpenSentinel1SafeRaster(std::string_view safe_path, std::string_view subswath,
                                           std::string_view polarisation) {
    constexpr std::string_view measurement_dir{"measurement"};
    if (!std::filesystem::exists(safe_path)) {
        throw std::invalid_argument("No file '" + std::string(safe_path) + "' found");
    }

    std::filesystem::path path(safe_path);
    std::string low_subswath = boost::to_lower_copy(std::string(subswath));
    std::string low_polarisation = boost::to_lower_copy(std::string(polarisation));
    std::string input_file{};

    if (common::zip::IsFileAnArchive(safe_path)) {
        boost::filesystem::path boost_path = std::string(safe_path);
        ceres::Zip dir(boost_path);
        // Convert path to SAFE
        auto leaf = path.filename();
        leaf.replace_extension("SAFE");

        const auto file_list = dir.List(leaf.string() + "/" + std::string(measurement_dir));
        const auto image_file =
            std::find_if(std::begin(file_list), std::end(file_list), [&low_subswath, &low_polarisation](auto& file) {
                return file.find(low_subswath) != std::string::npos && file.find(low_polarisation) != std::string::npos;
            });
        if (image_file == std::end(file_list)) {
            std::invalid_argument("SAFE does not contain raster for '" + low_subswath + "' and '" + low_polarisation +
                                  "'.");
        }
        input_file = leaf.string() + "/measurement/" + *image_file;
        return std::make_shared<T>(gdal::constants::GDAL_ZIP_PREFIX.data() + path.string() + "/" + input_file);
    } else {
        std::filesystem::path measurement = path.string() + "/" + std::string(measurement_dir);
        std::filesystem::directory_iterator end_itr;
        for (std::filesystem::directory_iterator itr(measurement); itr != end_itr; itr++) {
            if (std::filesystem::is_regular_file(itr->path())) {
                std::string current_file = itr->path().string();
                // Also check that file ends with .tif, since gdalinfo etc. could produce a metadata file ending aux.xml
                if (current_file.find(low_subswath) != std::string::npos &&
                    current_file.find(low_polarisation) != std::string::npos &&
                    current_file.find(".tif", current_file.length() - 5) != std::string::npos) {
                    LOGV << "Selecting tif for reading: " << current_file;
                    return std::make_shared<T>(current_file);
                }
            }
        }
    }

    throw std::invalid_argument("Could not find raster for subswath '" + std::string(subswath) +
                                "' and polarisation '" + std::string(polarisation) + "' in " + std::string(safe_path));
}
}  // namespace alus::dataset