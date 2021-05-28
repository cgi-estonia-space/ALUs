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
#include <cstddef>
#include <stdexcept>
#include <string_view>
#include <vector>

#include "gdal_tile_writer.h"

namespace alus {

GdalTileWriter::GdalTileWriter(std::string_view file_name, const BandParams& band_params,
                               const std::vector<double>& affine_geo_transform_out,
                               std::string_view data_projection_out)
    : IDataTileWriter(file_name, band_params, affine_geo_transform_out, data_projection_out), do_close_dataset_{true} {
    auto* const po_driver = GetGDALDriverManager()->GetDriverByName("GTiff");
    CHECK_GDAL_PTR(po_driver);
    InitializeOutputDataset(po_driver, GetGeoTransform(), GetDataProjection());
}

GdalTileWriter::GdalTileWriter(GDALDriver* output_driver, const BandParams& band_params,
                               const std::vector<double>& affine_geo_transform_out,
                               std::string_view data_projection_out)
    : IDataTileWriter("", band_params, affine_geo_transform_out, data_projection_out), do_close_dataset_{false} {
    CHECK_GDAL_PTR(output_driver);
    InitializeOutputDataset(output_driver, GetGeoTransform(), GetDataProjection());
}

void GdalTileWriter::WriteTile(const Tile& tile, float* tile_data, std::size_t tile_data_size) {
    if (tile_data_size > static_cast<std::size_t>(GetBandCount() * tile.GetXSize() * tile.GetYSize())) {
        throw std::runtime_error("tile data buffer overflow");
    }
    CHECK_GDAL_ERROR(output_dataset_->RasterIO(GF_Write, tile.GetXMin(), tile.GetYMin(), tile.GetXSize(),
                                               tile.GetYSize(), tile_data, tile.GetXSize(), tile.GetYSize(),
                                               GDALDataType::GDT_Float32, GetBandCount(), GetBandMap(), 0, 0, 0));
}

void GdalTileWriter::CloseDataSet() {
    if (output_dataset_ && do_close_dataset_) {
        GDALClose(output_dataset_);
        output_dataset_ = nullptr;
    }
}

void GdalTileWriter::InitializeOutputDataset(GDALDriver* output_driver, std::vector<double>& affine_geo_transform_out,
                                             std::string_view data_projection_out) {
    output_dataset_ =
        output_driver->Create(GetFileName().data(), GetBandXSize(), GetBandYSize(), 1, GDT_Float32, nullptr);
    CHECK_GDAL_PTR(output_dataset_);
    if (!GetGeoTransform().empty()) {
        CHECK_GDAL_ERROR(output_dataset_->SetGeoTransform(affine_geo_transform_out.data()));
    }
    CHECK_GDAL_ERROR(output_dataset_->SetProjection(data_projection_out.data()));
}

GdalTileWriter::~GdalTileWriter() { CloseDataSet(); }

}  // namespace alus
