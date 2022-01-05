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

#include <cstddef>
#include <string_view>
#include <vector>

#include <gdal_priv.h>

#include "band_params.h"
#include "gdal_util.h"
#include "i_data_tile_writer.h"
#include "tile.h"

namespace alus {
namespace coherence_cuda {
class GdalTileWriter : public IDataTileWriter {
private:
    GDALDataset* output_dataset_{};
    bool do_close_dataset_;

    void InitializeOutputDataset(GDALDriver* output_driver, std::vector<double>& affine_geo_transform_out,
                                 std::string_view data_projection_out);

public:
    GdalTileWriter(std::string_view file_name, const BandParams& band_params,
                   const std::vector<double>& affine_geo_transform_out, std::string_view data_projection_out);
    GdalTileWriter(GDALDriver* output_driver, const BandParams& band_params,
                   const std::vector<double>& affine_geo_transform_out, std::string_view data_projection_out);
    GdalTileWriter(const GdalTileWriter&) = delete;
    GdalTileWriter& operator=(const GdalTileWriter&) = delete;
    ~GdalTileWriter() override;
    //    void WriteTile(const Tile& tile, void* tile_data, std::size_t tile_data_size) override;
    void WriteTile(const Tile& tile, float* tile_data, std::size_t tile_data_size) override;
    [[nodiscard]] GDALDataset* GetGdalDataset() const { return output_dataset_; }
    void CloseDataSet();
};
}  // namespace coherence-cuda
}  // namespace alus
