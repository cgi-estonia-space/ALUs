#include "gdal_tile_reader.h"

#include <utility>

#include "gdal_util.h"
#include "i_data_tile_read_write_base.h"

namespace alus {

GdalTileReader::GdalTileReader(const std::string_view file_name, std::vector<int> band_map, int band_count,
                               bool has_transform)
    : IDataTileReader(file_name, std::move(band_map), band_count) {
    GDALAllRegister();
    dataset_ = static_cast<GDALDataset*>(GDALOpen(file_name.data(), GA_ReadOnly));
    CHECK_GDAL_PTR(dataset_);
    band_count_ = dataset_->GetRasterCount();
    band_x_size_ = dataset_->GetRasterXSize();
    band_y_size_ = dataset_->GetRasterYSize();
    data_projection_ = dataset_->GetProjectionRef();
    if (has_transform) {
        affine_geo_transform_.resize(6);
        CHECK_GDAL_ERROR(dataset_->GetGeoTransform(affine_geo_transform_.data()));
    }
}

void GdalTileReader::ReadTile(const Tile& tile) {
    AllocateForTileData(tile);
    CHECK_GDAL_ERROR(dataset_->RasterIO(GF_Read, tile.GetXMin(), tile.GetYMin(), tile.GetXSize(), tile.GetYSize(),
                                        data_, tile.GetXSize(), tile.GetYSize(), GDALDataType::GDT_Float32, band_count_,
                                        band_map_.data(), 0, 0, 0));
}

void GdalTileReader::AllocateForTileData(const Tile& tile) {
    CleanBuffer();
    data_ = static_cast<float*>(
        CPLMalloc(sizeof(GDALDataType::GDT_Float32) * tile.GetXSize() * tile.GetYSize() * band_count_));
}

void GdalTileReader::CleanBuffer() {
    if (data_) {
        CPLFree(data_);
        data_ = nullptr;
    }
}
void GdalTileReader::CloseDataSet() {
    if (dataset_) {
        GDALClose(dataset_);
        dataset_ = nullptr;
    }
}

float* GdalTileReader::GetData() const { return data_; }

const std::string_view GdalTileReader::GetDataProjection() const { return data_projection_; }

std::vector<double> GdalTileReader::GetGeoTransform() const { return affine_geo_transform_; }

// todo: support multiple bands and different data types?
double GdalTileReader::GetValueAtXy(int x, int y) const {
    int index = band_x_size_ * y + x;
    return data_[index];
}

GdalTileReader::~GdalTileReader() {
    CleanBuffer();
    CloseDataSet();
}

}  // namespace alus