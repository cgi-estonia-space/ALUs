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
#include "gdal_tile_writer.h"

namespace alus {

GdalTileWriter::GdalTileWriter(const std::string_view file_name, std::vector<int> band_map, int& band_count,
                               const int& band_x_size, const int& band_y_size, int band_x_min, int band_y_min,
                               std::vector<double> affine_geo_transform_out, const std::string_view data_projection_out)
    : IDataTileWriter(file_name, band_map, band_count, band_x_size, band_y_size, band_x_min, band_y_min,
                      affine_geo_transform_out, data_projection_out), do_close_dataset_{true} {
    const auto po_driver = GetGDALDriverManager()->GetDriverByName("GTiff");
    CHECK_GDAL_PTR(po_driver);
    InitializeOutputDataset(po_driver, std::move(affine_geo_transform_out), data_projection_out);
}

GdalTileWriter::GdalTileWriter(GDALDriver* output_driver, std::vector<int> band_map, int& band_count,
                               const int& band_x_size, const int& band_y_size, int band_x_min, int band_y_min,
                               std::vector<double> affine_geo_transform_out, std::string_view data_projection_out)
    : IDataTileWriter("", band_map, band_count, band_x_size, band_y_size, band_x_min, band_y_min,
                      affine_geo_transform_out, data_projection_out), do_close_dataset_{false} {

    CHECK_GDAL_PTR(output_driver);
    InitializeOutputDataset(output_driver, std::move(affine_geo_transform_out), data_projection_out);
}

// todo: IDataTileOut
void GdalTileWriter::WriteTile(const Tile& tile, void* tile_data) {
    CHECK_GDAL_ERROR(output_dataset_->RasterIO(
        GF_Write, tile.GetXMin(), tile.GetYMin(), tile.GetXSize(), tile.GetYSize(), tile_data, tile.GetXSize(),
        tile.GetYSize(), GDALDataType::GDT_Float32, this->band_count_, this->band_map_.data(), 0, 0, 0));
}

void GdalTileWriter::CloseDataSet() {
    if (this->output_dataset_ && do_close_dataset_) {
        GDALClose(this->output_dataset_);
        this->output_dataset_ = nullptr;
    }
}

void GdalTileWriter::InitializeOutputDataset(GDALDriver* output_driver, std::vector<double> affine_geo_transform_out,
                                             const std::string_view data_projection_out) {
    this->output_dataset_ =
        output_driver->Create(file_name_.data(), band_x_size_, band_y_size_, 1, GDT_Float32, nullptr);
    CHECK_GDAL_PTR(this->output_dataset_);
    if (!affine_geo_transform_.empty()) {
        CHECK_GDAL_ERROR(output_dataset_->SetGeoTransform(affine_geo_transform_out.data()));
    }
    CHECK_GDAL_ERROR(output_dataset_->SetProjection(data_projection_out.data()));
}

GdalTileWriter::~GdalTileWriter() { this->CloseDataSet(); }

}  // namespace alus