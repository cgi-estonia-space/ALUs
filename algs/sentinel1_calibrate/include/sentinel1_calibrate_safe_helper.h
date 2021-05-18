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

#include <map>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include <gdal_priv.h>
#include <boost/filesystem/path.hpp>

namespace alus::sentinel1calibrate {

class Sentinel1CalibrateSafeHelper {
public:
    explicit Sentinel1CalibrateSafeHelper(std::string_view safe_directory_path);
    Sentinel1CalibrateSafeHelper(const Sentinel1CalibrateSafeHelper&) = delete;
    Sentinel1CalibrateSafeHelper(Sentinel1CalibrateSafeHelper&&) = delete;
    Sentinel1CalibrateSafeHelper& operator=(const Sentinel1CalibrateSafeHelper&) = delete;
    Sentinel1CalibrateSafeHelper& operator=(Sentinel1CalibrateSafeHelper&&) = delete;
    ~Sentinel1CalibrateSafeHelper();

    GDALDataset* GetSubDatasetByPolarisationAndSubSwath(std::string_view polarisation_and_sub_swath);

private:
    std::string safe_directory_path_;
    std::map<std::string, GDALDataset*, std::less<>> sub_datasets_{};
    std::vector<boost::filesystem::path> sub_dataset_paths_{};

    void ReadSubDatasets();
    [[nodiscard]] bool DatasetNameContains(std::string_view string, std::string_view key) const;
};
}  // namespace alus::sentinel1calibrate