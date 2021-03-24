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
    GdalTileReader(std::string_view file_name, const std::vector<int>& band_map, int band_count, bool has_transform);
    GdalTileReader(GDALDataset* dataset, const std::vector<int>& band_map, int band_count, bool has_transform);
    GdalTileReader(const GdalTileReader&) = delete;
    GdalTileReader& operator=(const GdalTileReader&) = delete;
    ~GdalTileReader() override;
    void ReadTile(const Tile& tile) override;
    void CloseDataSet();
    [[nodiscard]] const std::vector<float>& GetData() const override;
    [[nodiscard]] double GetValueAtXy(int x, int y) const override;
    // todo:  void ReadTileToTensors(const IDataTileIn &tile) override;
private:
    GDALDataset* dataset_{};
    bool do_close_dataset_;
    std::vector<float> data_{};

    void InitializeDatasetProperties(GDALDataset* dataset, bool has_transform);
};
}  // namespace alus
