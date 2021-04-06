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

#include "gdal_util.h"
#include "i_data_tile_writer.h"
#include "tile.h"

namespace alus {
class GdalTileWriter : virtual public IDataTileWriter {
private:

    GDALDataset* output_dataset_{};
    bool do_close_dataset_;

    void InitializeOutputDataset(GDALDriver* output_driver, std::vector<double> affine_geo_transform_out,
                                 const std::string_view data_projection_out);

public:
    GdalTileWriter(std::string_view file_name, std::vector<int> band_map, int& band_count, const int& band_x_size,
                   const int& band_y_size, int band_x_min, int band_y_min, std::vector<double> affine_geo_transform_out,
                   std::string_view data_projection_out);
    GdalTileWriter(GDALDriver* output_driver, std::vector<int> band_map, int& band_count, const int& band_x_size,
                   const int& band_y_size, int band_x_min, int band_y_min, std::vector<double> affine_geo_transform_out,
                   std::string_view data_projection_out);

    void WriteTile(const Tile& tile, void* tile_data) override;
    GDALDataset* GetGdalDataset() const { return output_dataset_; }
    void CloseDataSet();
    ~GdalTileWriter() override;
};
}  // namespace alus
