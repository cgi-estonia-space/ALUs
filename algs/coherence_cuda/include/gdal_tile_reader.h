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

#include <deque>
#include <mutex>
#include <string_view>
#include <vector>

#include <gdal_priv.h>

#include "tile.h"

namespace alus {
namespace coherence_cuda {
class GdalTileReader {
public:
    explicit GdalTileReader(std::string_view file_name);
    explicit GdalTileReader(const std::vector<GDALDataset*>& dataset);
    GdalTileReader(const GdalTileReader&) = delete;
    GdalTileReader& operator=(const GdalTileReader&) = delete;
    ~GdalTileReader();
    void ReadTile(const Tile& tile, float* data, int band_nr);
    void CloseDataSet();
    [[nodiscard]] int GetBandXSize() const { return datasets_.at(0)->GetRasterXSize(); }
    [[nodiscard]] int GetBandYSize() const { return datasets_.at(0)->GetRasterYSize(); }
    [[nodiscard]] int GetBandXMin() const { return 0; }
    [[nodiscard]] int GetBandYMin() const { return 0; }
    double GetValueAtXy(int x, int y);
    [[nodiscard]] std::string GetDataProjection() const { return data_projection_; }
    [[nodiscard]] const std::vector<double>& GetGeoTransform() const { return affine_geo_transform_; }
private:
    std::vector<GDALDataset*> datasets_;
    std::deque<std::mutex> mutexes_;
    std::string data_projection_;
    std::vector<double> affine_geo_transform_;
};
}  // namespace coherence_cuda
}  // namespace alus
