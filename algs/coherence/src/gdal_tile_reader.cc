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

#include <string_view>
#include <vector>

#include "alus_log.h"
#include "gdal_util.h"
#include "i_data_tile_read_write_base.h"

namespace alus {

GdalTileReader::GdalTileReader(const std::string_view file_name, const std::vector<int>& band_map, int band_count,
                               bool has_transform)
    : IDataTileReader(file_name, band_map, band_count), do_close_dataset_{true} {
    GDALAllRegister();
    dataset_ = static_cast<GDALDataset*>(GDALOpen(file_name.data(), GA_ReadOnly));
    CHECK_GDAL_PTR(dataset_);
    InitializeDatasetProperties(dataset_, has_transform);
}

GdalTileReader::GdalTileReader(GDALDataset* dataset, const std::vector<int>& band_map, int band_count,
                               bool has_transform)
    : IDataTileReader("", band_map, band_count), dataset_{dataset}, do_close_dataset_{false} {
    CHECK_GDAL_PTR(dataset_);
    InitializeDatasetProperties(dataset_, has_transform);
}

void GdalTileReader::ReadTile(const Tile& tile) {
    data_.resize(tile.GetXSize() * tile.GetYSize() * GetBandCount());
    CHECK_GDAL_ERROR(dataset_->RasterIO(GF_Read, tile.GetXMin(), tile.GetYMin(), tile.GetXSize(), tile.GetYSize(),
                                        data_.data(), tile.GetXSize(), tile.GetYSize(), GDALDataType::GDT_Float32,
                                        GetBandCount(), GetBandMap(), 0, 0, 0));
}

void GdalTileReader::CloseDataSet() {
    if (dataset_ && do_close_dataset_) {
        GDALClose(dataset_);
        dataset_ = nullptr;
    }
}

double GdalTileReader::GetValueAtXy(int x, int y) const { return data_.at(GetBandXSize() * y + x); }

void GdalTileReader::InitializeDatasetProperties(GDALDataset* dataset, bool has_transform) {
    SetBandCount(dataset->GetRasterCount());
    SetBandXSize(dataset->GetRasterXSize());
    SetBandYSize(dataset->GetRasterYSize());
    SetDataProjection(dataset->GetProjectionRef());
    if (has_transform) {
        GetGeoTransform().resize(6);
        const auto result = dataset->GetGeoTransform(GetGeoTransform().data());
        // Fetching transform has been requested, but dataset's transform is invalid.
        if (result != CE_None) {
            LOGI << "Geo transform parameters are missing in input dataset - " << GetFileName();
            GetGeoTransform().clear();
        }
    }
}

GdalTileReader::~GdalTileReader() { CloseDataSet(); }
const std::vector<float>& GdalTileReader::GetData() const { return data_; }

}  // namespace alus