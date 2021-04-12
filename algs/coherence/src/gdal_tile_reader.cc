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
#include "gdal_tile_reader.h"

#include <utility>

#include "gdal_util.h"
#include "i_data_tile_read_write_base.h"

#include <iostream>

namespace alus {

GdalTileReader::GdalTileReader(const std::string_view file_name, std::vector<int> band_map, int band_count,
                               bool has_transform)
    : IDataTileReader(file_name, std::move(band_map), band_count), do_close_dataset_{true} {
    GDALAllRegister();
    dataset_ = static_cast<GDALDataset*>(GDALOpen(file_name.data(), GA_ReadOnly));
    CHECK_GDAL_PTR(dataset_);
    InitializeDatasetProperties(dataset_, has_transform);
}

GdalTileReader::GdalTileReader(GDALDataset* dataset, std::vector<int> band_map, int band_count, bool has_transform)
    : IDataTileReader("", std::move(band_map), band_count), dataset_{dataset}, do_close_dataset_{false} {
    CHECK_GDAL_PTR(dataset_);
    InitializeDatasetProperties(dataset_, has_transform);
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
    if (dataset_ && do_close_dataset_) {
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

void GdalTileReader::InitializeDatasetProperties(GDALDataset* dataset, bool has_transform) {
    band_count_ = dataset->GetRasterCount();
    band_x_size_ = dataset->GetRasterXSize();
    band_y_size_ = dataset->GetRasterYSize();
    data_projection_ = dataset->GetProjectionRef();
    if (has_transform) {
        affine_geo_transform_.resize(6);
        const auto result = dataset->GetGeoTransform(affine_geo_transform_.data());
        // Fetching transform has been requested, but dataset's transform is invalid.
        if (result != CE_None) {
            // TODO: Use logging system to log this message.
            std::cout << "Geo transform parameters are missing in input dataset - " << file_name_ << std::endl;
            this->affine_geo_transform_.clear();
        }
    }
}

GdalTileReader::~GdalTileReader() {
    CleanBuffer();
    CloseDataSet();
}

}  // namespace alus