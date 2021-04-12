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

#include <string_view>
#include <vector>

#include <gdal_priv.h>

#include "i_data_tile_read_write_base.h"
#include "i_data_tile_reader.h"
#include "tile.h"

namespace alus {
class GdalTileReader : public IDataTileReader {
public:
    GdalTileReader(std::string_view file_name, std::vector<int> band_map, int band_count, bool has_transform);
    GdalTileReader(GDALDataset* dataset, std::vector<int> band_map, int band_count, bool has_transform);
    GdalTileReader(const GdalTileReader&) = delete;
    GdalTileReader& operator=(const GdalTileReader&) = delete;
    ~GdalTileReader() override;

    void ReadTile(const Tile& tile) override;
    void CleanBuffer();
    void CloseDataSet();
    [[nodiscard]] float* GetData() const override;
    [[nodiscard]] int GetBandXSize() const override { return band_x_size_; }
    [[nodiscard]] int GetBandYSize() const override { return band_y_size_; }
    [[nodiscard]] int GetBandXMin() const override { return band_x_min_; }
    [[nodiscard]] int GetBandYMin() const override { return band_y_min_; }

    [[nodiscard]] const std::string_view GetDataProjection() const override;
    [[nodiscard]] std::vector<double> GetGeoTransform() const override;
    [[nodiscard]] double GetValueAtXy(int x, int y) const override;
    // todo:  void ReadTileToTensors(const IDataTileIn &tile) override;
private:
    void AllocateForTileData(const Tile& tile);
    void InitializeDatasetProperties(GDALDataset* dataset, bool has_transform);

    GDALDataset* dataset_{};
    bool do_close_dataset_;
    float* data_{};
};
}  // namespace alus
