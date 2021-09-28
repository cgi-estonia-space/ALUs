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
#include "target_dataset.h"

#include <cstdint>

#include "gdal_util.h"

namespace alus {

template <typename OutputType>
TargetDataset<OutputType>::TargetDataset(TargetDatasetParams params)
    : dataset_per_band_{params.dataset_per_band}, dimensions{params.dimension} {
    auto const driver = params.driver;
    CHECK_GDAL_PTR(driver);

    gdal_data_type_ = FindGdalDataType<OutputType>();

    const size_t n_datasets = dataset_per_band_ ? params.band_count : 1;
    const int bands_per_dataset = dataset_per_band_ ? 1 : params.band_count;
    datasets_.resize(n_datasets);
    mutexes_.resize(n_datasets);

    try {
        for (auto*& dataset : datasets_) {
            dataset = driver->Create(params.filename.data(), dimensions.columnsX, dimensions.rowsY, bands_per_dataset,
                                     gdal_data_type_, nullptr);
            CHECK_GDAL_PTR(dataset);
            if (dataset) {
                double gt[GeoTransformConstruct::GDAL_GEOTRANSFORM_PARAMETERS_LENGTH];
                std::copy_n(params.transform, GeoTransformConstruct::GDAL_GEOTRANSFORM_PARAMETERS_LENGTH, gt);
                CHECK_GDAL_ERROR(dataset->SetGeoTransform(gt));
                CHECK_GDAL_ERROR(dataset->SetProjection(params.projectionRef));
            }
        }
    } catch (...) {
        for (auto* dataset : datasets_) {
            GDALClose(dataset);
        }
        throw;
    }
}

template <typename OutputType>
void TargetDataset<OutputType>::WriteRectangle(OutputType* from, Rectangle area, int band_nr) {
    if (area.width <= 0 || area.height <= 0) {
        throw std::invalid_argument(std::to_string(area.width) + " and " + std::to_string(area.height) +
                                    " are not acceptable parameters for width and height.");
    }

    GDALRasterBand* band = nullptr;
    std::mutex* mutex = nullptr;
    if (dataset_per_band_) {
        band = datasets_.at(band_nr - 1)->GetRasterBand(1);
        mutex = &mutexes_.at(band_nr - 1);
    } else {
        band = datasets_.at(0)->GetRasterBand(band_nr);
        mutex = &mutexes_.at(0);
    }

    std::unique_lock lock(*mutex);
    CHECK_GDAL_ERROR(band->RasterIO(GF_Write, area.x, area.y, area.width, area.height, from, area.width, area.height,
                                    gdal_data_type_, 0, 0));
}

template <typename OutputType>
TargetDataset<OutputType>::~TargetDataset() {
    for (auto* dataset : datasets_) {
        GDALClose(dataset);
    }
}

template class TargetDataset<double>;
template class TargetDataset<float>;
template class TargetDataset<int16_t>;
template class TargetDataset<int>;

}  // namespace alus