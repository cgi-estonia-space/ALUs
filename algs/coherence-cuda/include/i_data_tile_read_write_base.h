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
#include <utility>
#include <vector>

#include "band_params.h"
#include "tile.h"

namespace alus {
class IDataTileReadWriteBase {
private:
    std::string file_name_;
    BandParams band_params_;
    std::string data_projection_;
    std::vector<double> affine_geo_transform_;

public:
    IDataTileReadWriteBase(std::string_view file_name, const std::vector<int>& band_map, int band_count)
        : file_name_(file_name), band_params_{band_map, band_count} {}
    IDataTileReadWriteBase(std::string_view file_name, BandParams band_params, std::string_view data_projection,
                           std::vector<double> affine_geo_transform)
        : file_name_{file_name},
          band_params_(std::move(band_params)),
          data_projection_{data_projection},
          affine_geo_transform_{std::move(affine_geo_transform)} {};
    IDataTileReadWriteBase(const IDataTileReadWriteBase&) = delete;
    IDataTileReadWriteBase& operator=(const IDataTileReadWriteBase&) = delete;
    [[nodiscard]] int GetBandXSize() const { return band_params_.band_x_size; }
    [[nodiscard]] int GetBandYSize() const { return band_params_.band_y_size; }
    [[nodiscard]] int GetBandXMin() const { return band_params_.band_x_min; }
    [[nodiscard]] int GetBandYMin() const { return band_params_.band_y_min; }
    [[nodiscard]] std::string GetDataProjection() const { return data_projection_; }
    [[nodiscard]] const std::vector<double>& GetGeoTransform() const { return affine_geo_transform_; }
    [[nodiscard]] std::vector<double>& GetGeoTransform() { return affine_geo_transform_; }
    [[nodiscard]] int GetBandCount() const { return band_params_.band_count; }
    [[nodiscard]] const std::vector<int>& GetBandMap() const { return band_params_.band_map; }
    [[nodiscard]] int* GetBandMap() { return band_params_.band_map.data(); }
    [[nodiscard]] const std::string& GetFileName() const { return file_name_; }
    void SetBandMap(const std::vector<int>& band_map) { band_params_.band_map = band_map; }
    void SetBandCount(int band_count) { band_params_.band_count = band_count; }
    void SetBandXSize(int band_x_size) { band_params_.band_x_size = band_x_size; }
    void SetBandYSize(int band_y_size) { band_params_.band_y_size = band_y_size; }
    void SetDataProjection(std::string_view data_projection) { data_projection_ = data_projection; }
};
}  // namespace alus
