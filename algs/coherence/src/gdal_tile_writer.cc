#include "gdal_tile_writer.h"

namespace alus {

GdalTileWriter::GdalTileWriter(const std::string_view file_name,
                               std::vector<int> band_map,
                               int &band_count,
                               const int &band_x_size,
                               const int &band_y_size,
                               int band_x_min,
                               int band_y_min,
                               std::vector<double> affine_geo_transform_out,
                               const std::string_view data_projection_out)
    : IDataTileWriter(file_name,
                      band_map,
                      band_count,
                      band_x_size,
                      band_y_size,
                      band_x_min,
                      band_y_min,
                      affine_geo_transform_out,
                      data_projection_out) {
    auto const po_driver = GetGDALDriverManager()->GetDriverByName("GTiff");
    CHECK_GDAL_PTR(po_driver);
    this->output_dataset_ = po_driver->Create(file_name.data(), band_x_size, band_y_size, 1, GDT_Float32, nullptr);
    CHECK_GDAL_PTR(this->output_dataset_);
    if (!affine_geo_transform_.empty()) {
        CHECK_GDAL_ERROR(output_dataset_->SetGeoTransform(affine_geo_transform_out.data()));
    }
    CHECK_GDAL_ERROR(output_dataset_->SetProjection(data_projection_out.data()));
}

// todo: IDataTileOut
void GdalTileWriter::WriteTile(const Tile &tile, void *tile_data) {
    CHECK_GDAL_ERROR(output_dataset_->RasterIO(GF_Write,
                                               tile.GetXMin(),
                                               tile.GetYMin(),
                                               tile.GetXSize(),
                                               tile.GetYSize(),
                                               tile_data,
                                               tile.GetXSize(),
                                               tile.GetYSize(),
                                               GDALDataType::GDT_Float32,
                                               this->band_count_,
                                               this->band_map_.data(),
                                               0,
                                               0,
                                               0));
}

void GdalTileWriter::CloseDataSet() {
    if (this->output_dataset_) {
        GDALClose(this->output_dataset_);
        this->output_dataset_ = nullptr;
    }
}

GdalTileWriter::~GdalTileWriter() { this->CloseDataSet(); }

}  // namespace alus