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

#include <string>
#include <string_view>
#include <unordered_map>

#include "dataset.h"
#include "raster_properties.h"
#include "sentinel2_constants.h"

namespace alus::resample {
class Sentinel2Dataset final {
public:
    Sentinel2Dataset() = delete;
    explicit Sentinel2Dataset(std::string_view dataset_path);

    size_t GetBandCount() const { return band_table_.size(); }
    RasterDimension GetBandDimension(size_t index) { return GetBandDimension(sentinel2::BANDS.at(index)); }
    RasterDimension GetBandDimension(std::string_view band_id) {
        return datasets_.at(band_table_.at(std::string(band_id)).dataset_index).GetRasterDimensions();
    }
    Dataset<uint16_t>& GetDataset(size_t index) { return GetDataset(sentinel2::BANDS.at(index)); }
    Dataset<uint16_t>& GetDataset(std::string_view band_id) {
        return datasets_.at(band_table_.at(std::string(band_id)).dataset_index);
    }
    size_t GetBandNoInDataset(size_t index);
    const std::vector<uint16_t>& GetBandData(size_t index) { return GetBandData(sentinel2::BANDS.at(index)); }
    const std::vector<uint16_t>& GetBandData(std::string_view band_id);
    std::string GetGranuleImgFilenameStemFor(std::string_view band_id) const {
        return granule_img_filename_stem_ + std::string(band_id);
    }
    std::string GetGranuleImgFilenameStemFor(size_t band_index) const {
        return GetGranuleImgFilenameStemFor(sentinel2::BANDS.at(band_index));
    }

private:
    void FetchBands(const std::vector<std::string>& subdataset_list);
    static std::vector<std::string> GetSubdatasetList(char** entries, size_t count);
    static void RemoveTrueColorImageDatasetFrom(std::vector<std::string>& subdataset_list);
    static std::string GetGranuleImgFilenameStem(const char* metadata_value);

    std::vector<Dataset<uint16_t>> datasets_;

    struct BandHandle {
        size_t dataset_index;
        size_t band_index_in_dataset;
        std::unordered_map<std::string, std::string> metadata;
    };
    std::unordered_map<std::string, BandHandle> band_table_;
    std::string granule_img_filename_stem_{};
};
}  // namespace alus::resample