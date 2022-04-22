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
#include "custom/gdal_image_reader.h"

#include <cstddef>
#include <stdexcept>

#include "alus_log.h"
#include "gdal_util.h"

namespace alus::snapengine::custom {

void GdalImageReader::ReadSubSampledData(const custom::Rectangle& rectangle, int band_indx) {

    // Sometimes SNAP might request to read data which is out of image bounds. This should resolve this issue.
    const auto width = rectangle.x + rectangle.width > dataset_->GetRasterXSize()
                           ? dataset_->GetRasterXSize() - rectangle.x
                           : rectangle.width;
    const auto height = rectangle.y + rectangle.height > dataset_->GetRasterYSize()
                            ? dataset_->GetRasterYSize() - rectangle.y
                            : rectangle.height;

    if (data_.size() != static_cast<std::size_t>(width) * static_cast<size_t>(height)) {
        data_.resize(static_cast<size_t>(width * height));
    }
    CHECK_GDAL_ERROR(dataset_->GetRasterBand(band_indx)->RasterIO(GF_Read, rectangle.x, rectangle.y, width, height,
                                                                  data_.data(), width, height,
                                                                  GDALDataType::GDT_Float32, 0, 0));
}

void GdalImageReader::Open(std::string_view path_to_band_file, bool has_transform, bool has_correct_proj) {
    file_ = path_to_band_file;
    dataset_ = static_cast<GDALDataset*>(GDALOpen(file_.c_str(), GA_ReadOnly));
    CHECK_GDAL_PTR(dataset_);
    InitializeDatasetProperties(dataset_, has_transform, has_correct_proj);
}

void GdalImageReader::TakeExternalDataset(GDALDataset* dataset) {
    CHECK_GDAL_PTR(dataset);
    dataset_ = dataset;
}

GdalImageReader::~GdalImageReader() {
    if (dataset_) {
        GDALClose(dataset_);
        dataset_ = nullptr;
    }
}

std::string GdalImageReader::GetDataProjection() const { return data_projection_; }
std::vector<double> GdalImageReader::GetGeoTransform() const { return affine_geo_transform_; }

void GdalImageReader::Close() {
    if (dataset_) {
        GDALClose(dataset_);
        dataset_ = nullptr;
    }
}

void GdalImageReader::InitializeDatasetProperties(GDALDataset* dataset, bool has_transform, bool has_correct_proj) {
    if (has_correct_proj) {
        data_projection_ = dataset->GetProjectionRef();
    }
    if (has_transform) {
        affine_geo_transform_.resize(gdal::constants::GDAL_GEOTRANSFORM_PARAMETER_COUNT);
        const auto result = dataset->GetGeoTransform(affine_geo_transform_.data());
        if (result != CE_None) {
            // TODO: Use logging system to log this message.  // NOLINT
            LOGW << "Geo transform parameters are missing in input dataset - " << file_;
            affine_geo_transform_.clear();
        }
    }
}

void GdalImageReader::ReadSubSampledData(const std::shared_ptr<custom::Rectangle>& rectangle,
                                         std::vector<int32_t>& data) {
    // todo:    later add support for subsampled data, this will change parameters for this function
    if (data.size() != static_cast<std::size_t>(rectangle->width) * static_cast<size_t>(rectangle->height)) {
        data.resize(static_cast<size_t>(rectangle->width) * static_cast<size_t>(rectangle->height));
    }

    CHECK_GDAL_ERROR(dataset_->RasterIO(GF_Read, rectangle->x, rectangle->y, rectangle->width, rectangle->height,
                                        data.data(), rectangle->width, rectangle->height, GDALDataType::GDT_Int32, 1,
                                        nullptr, 0, 0, 0));
}
void GdalImageReader::ReleaseDataset() { dataset_ = nullptr; }
void GdalImageReader::ReadSubSampledData(const alus::Rectangle& rectangle, int band_indx) {
    custom::Rectangle region(rectangle);
    ReadSubSampledData(region, band_indx);
}

}  // namespace alus::snapengine::custom