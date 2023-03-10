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

#include <iostream>
#include <string_view>
#include <vector>

#include <gdal.h>

#include "gdal_util.h"

namespace alus {
namespace coherence_cuda {
GdalTileReader::GdalTileReader(const std::string_view file_name) {
    auto* dataset = static_cast<GDALDataset*>(GDALOpen(file_name.data(), GA_ReadOnly));
    CHECK_GDAL_PTR(dataset);
    mutexes_.resize(1);
    datasets_.push_back(dataset);
    affine_geo_transform_.resize(6);
    if (dataset->GetGeoTransform(affine_geo_transform_.data()) != CE_None) {
        affine_geo_transform_.clear();
    }
    data_projection_ = dataset->GetProjectionRef();
}

GdalTileReader::GdalTileReader(const std::vector<GDALDataset*>& datasets) : datasets_{datasets} {
    mutexes_.resize(datasets_.size());
}

void GdalTileReader::ReadTile(const Tile& tile, float* data, int band_nr) {
    std::mutex* mutex = nullptr;
    GDALRasterBand* band = nullptr;

    if (datasets_.size() == 1) {
        mutex = &mutexes_.at(0);
        band = datasets_.at(0)->GetRasterBand(band_nr);
    } else {
        mutex = &mutexes_.at(band_nr - 1);
        band = datasets_.at(band_nr - 1)->GetRasterBand(1);
    }

    std::unique_lock lock(*mutex);
    printf("band = %d %d -> read = (%d %d %d %d)\n", band->GetXSize(), band->GetYSize(),tile.GetXMin(), tile.GetYMin(), tile.GetXSize(), tile.GetYSize());
    CHECK_GDAL_ERROR(band->RasterIO(GF_Read, tile.GetXMin(), tile.GetYMin(), tile.GetXSize(), tile.GetYSize(), data,
                                    tile.GetXSize(), tile.GetYSize(), GDALDataType::GDT_Float32, 0, 0));
}

void GdalTileReader::CloseDataSet() {
    for (auto*& dataset : datasets_) {
        GDALClose(dataset);
    }
    datasets_.clear();
}

double GdalTileReader::GetValueAtXy(int x, int y) {
    // TODO refactor, this function is only needed for coherence integration test, when reading beam dimap
    float val = 0;
    CHECK_GDAL_ERROR(datasets_.at(0)->GetRasterBand(1)->RasterIO(GF_Read, x, y, 1, 1, &val, 1, 1, GDT_Float32, 0, 0));
    return val;
}

GdalTileReader::~GdalTileReader() { CloseDataSet(); }

}  // namespace coherence-cuda
}  // namespace alus