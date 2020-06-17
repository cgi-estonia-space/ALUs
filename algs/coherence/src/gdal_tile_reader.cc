#include "gdal_tile_reader.h"
#include "i_data_tile_read_write_base.h"

namespace alus {

GdalTileReader::GdalTileReader(const std::string_view file_name,
                               std::vector<int> band_map,
                               int &band_count,
                               bool has_transform)
    : IDataTileReader(file_name, band_map, band_count, has_transform) {
    GDALAllRegister();
    this->dataset_ = (GDALDataset *)GDALOpen(file_name.data(), GA_ReadOnly);
    CHECK_GDAL_PTR(this->dataset_);
    this->band_count_ = dataset_->GetRasterCount();
    this->band_x_size_ = dataset_->GetRasterXSize();
    this->band_y_size_ = dataset_->GetRasterYSize();
    this->data_projection_ = dataset_->GetProjectionRef();
    if (has_transform) {
        this->affine_geo_transform_.resize(6);
        CHECK_GDAL_ERROR(this->dataset_->GetGeoTransform(this->affine_geo_transform_.data()));
    }
}

void GdalTileReader::ReadTile(const Tile &tile) {
    this->AllocateForTileData(tile);
    CHECK_GDAL_ERROR(this->dataset_->RasterIO(GF_Read,
                                              tile.GetXMin(),
                                              tile.GetYMin(),
                                              tile.GetXSize(),
                                              tile.GetYSize(),
                                              this->data_,
                                              tile.GetXSize(),
                                              tile.GetYSize(),
                                              GDALDataType::GDT_Float32,
                                              this->band_count_,
                                              this->band_map_.data(),
                                              0,
                                              0,
                                              0));
}

void GdalTileReader::AllocateForTileData(const Tile &tile) {
    this->data_ =
        (float *)CPLMalloc(sizeof(GDALDataType::GDT_Float32) * tile.GetXSize() * tile.GetYSize() * this->band_count_);
}

void GdalTileReader::CleanBuffer() {
    if (this->data_) {
        CPLFree(this->data_);
        this->data_ = nullptr;
    }
}
void GdalTileReader::CloseDataSet() {
    if (this->dataset_) {
        GDALClose(this->dataset_);
        this->dataset_ = nullptr;
    }
}

float *GdalTileReader::GetData() const { return this->data_; }

const std::string_view GdalTileReader::GetDataProjection() const { return this->data_projection_; }

std::vector<double> GdalTileReader::GetGeoTransform() const { return this->affine_geo_transform_; }

// todo: support multiple bands and different data types?
double GdalTileReader::GetValueAtXy(int x, int y) const {
    int index = this->band_x_size_ * y + x;
    return this->data_[index];
}

GdalTileReader::~GdalTileReader() {
    this->CleanBuffer();
    this->CloseDataSet();
}

}  // namespace alus