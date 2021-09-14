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
#include "sentinel1_calibrate_safe_helper.h"

#include <memory>
#include <string>
#include <string_view>

#include <gdal.h>
#include <gdal_priv.h>
#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>

#include "dataset.h"

namespace alus::sentinel1calibrate {
void Sentinel1CalibrateSafeHelper::ReadSubDatasets() {
    const auto sub_dataset_dir = safe_directory_path_ + "/measurement";
    for (const auto& file : boost::filesystem::directory_iterator(sub_dataset_dir)) {
        sub_dataset_paths_.push_back(file.path());
    }
}
Sentinel1CalibrateSafeHelper::Sentinel1CalibrateSafeHelper(std::string_view safe_directory_path)
    : safe_directory_path_(safe_directory_path) {
    ReadSubDatasets();
}

GDALDataset* Sentinel1CalibrateSafeHelper::GetSubDatasetByPolarisationAndSubSwath(
    std::string_view polarisation_and_sub_swath) {
    auto does_dataset_map_contain_key = [&](std::string_view key) {
        return sub_datasets_.find(key.data()) != sub_datasets_.end();
    };

    for (const auto& path : sub_dataset_paths_) {
        const auto dataset_name = path.filename().string();
        if (DatasetNameContains(dataset_name, polarisation_and_sub_swath)) {
            // Check if dataset was already opened
            if (does_dataset_map_contain_key(dataset_name)) {
                return sub_datasets_.at(dataset_name);
            }
            auto* sub_dataset = static_cast<GDALDataset*>(GDALOpen(path.c_str(), GA_ReadOnly));
            sub_datasets_.try_emplace(dataset_name, sub_dataset);
            return sub_datasets_.at(dataset_name);
        }
    }
    return nullptr;
}

bool Sentinel1CalibrateSafeHelper::DatasetNameContains(std::string_view string, std::string_view key) const {
    const auto lower_key = boost::to_lower_copy(std::string(key));
    std::vector<std::string> tokens;
    boost::split(tokens, lower_key, boost::is_any_of("_"), boost::token_compress_on);

    return std::all_of(tokens.begin(), tokens.end(),
                       [&string](std::string_view sub_key) { return string.find(sub_key) != std::string::npos; });
}
Sentinel1CalibrateSafeHelper::~Sentinel1CalibrateSafeHelper() {
    for (const auto& [dataset_name, dataset] : sub_datasets_) {
        GDALClose(dataset);
        (void)dataset_name;
    }
}
}  // namespace alus::sentinel1calibrate
