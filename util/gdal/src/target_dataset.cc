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

#include <iostream>
#include <cstdint>

#include "gdal_util.h"

namespace alus {

template<typename OutputType>
TargetDataset<OutputType>::TargetDataset(TargetDatasetParams params)
    : dimensions{params.dimension} {
    auto const driver = params.driver;
    CHECK_GDAL_PTR(driver);

    gdal_data_type_ = FindGdalDataType<OutputType>();

    this->gdalDs = driver->Create(params.filename.data(), dimensions.columnsX, dimensions.rowsY, params.band_count, gdal_data_type_, nullptr);
    CHECK_GDAL_PTR(this->gdalDs);
    double gt[GeoTransformConstruct::GDAL_GEOTRANSFORM_PARAMETERS_LENGTH];
    //auto inputGt = dsWithProperties.GetTransform();
    std::copy_n(params.transform, GeoTransformConstruct::GDAL_GEOTRANSFORM_PARAMETERS_LENGTH, gt);
    CHECK_GDAL_ERROR(this->gdalDs->SetGeoTransform(gt));
    CHECK_GDAL_ERROR(this->gdalDs->SetProjection(params.projectionRef));

}

template<typename OutputType>
void TargetDataset<OutputType>::WriteRectangle(OutputType *from, Rectangle area, int band_nr) {
    if (area.width <= 0 || area.height <= 0) {
        throw std::invalid_argument(std::to_string(area.width) +" and " + std::to_string(area.height) + " are not acceptable parameters for width and height.");
    }

    CHECK_GDAL_ERROR(this->gdalDs->GetRasterBand(band_nr)->RasterIO(
        GF_Write, area.x, area.y, area.width, area.height, from, area.width, area.height, gdal_data_type_, 0, 0));
}

template<typename OutputType>
TargetDataset<OutputType>::~TargetDataset() {
    if (this->gdalDs) {
        GDALClose(this->gdalDs);
        this->gdalDs = nullptr;
    }
}

template class TargetDataset<double>;
template class TargetDataset<float>;
template class TargetDataset<int16_t>;
template class TargetDataset<int>;

}  // namespace alus