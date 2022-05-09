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

#include <memory>
#include <string>
#include <string_view>

#include <gdal_priv.h>

#include "raster_properties.h"
#include "sentinel2_dataset.h"

namespace alus::resample {

class DatasetRegister final {
public:
    DatasetRegister() = delete;
    explicit DatasetRegister(std::string_view dataset_path);

    [[nodiscard]] RasterDimension GetBandDimension(size_t band_index);
    [[nodiscard]] size_t GetBandCount();
    [[nodiscard]] std::string GetBandDescription(size_t band_index) const;
    [[nodiscard]] GDALDataType GetDataType(size_t band_index);
    [[nodiscard]] size_t GetRasterDataTypeSize(size_t band_index);
    std::unique_ptr<uint8_t[]> GetBandData(size_t band_index, size_t& buffer_size_bytes);
    [[nodiscard]] GeoTransformParameters GetGeoTransform(size_t band_index);
    [[nodiscard]] OGRSpatialReference GetSrs() const;
    [[nodiscard]] std::string GetGranuleImageFilenameStemFor(size_t band_index) const;
    GDALDriver* GetGdalDriver();
    std::vector<std::pair<std::string, std::pair<std::string, std::string>>> GetBandMetadata(size_t band_index);
    bool GetBandNoDataValue(size_t band_index, double& no_data_value);

    ~DatasetRegister();

private:
    std::string input_path_;
    bool is_sentinel2_dataset_{false};
    std::unique_ptr<Sentinel2Dataset> s2_ds_{};
    GDALDataset* generic_ds_{};
};

}  // namespace alus::resample