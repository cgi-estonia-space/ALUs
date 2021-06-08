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
#include "dataset.h"

#include <cstdint>
#include <map>

#include "alus_log.h"
#include "gdal.h"
#include "gdal_util.h"

namespace {
const std::map<alus::GeoTransformSourcePriority, std::string> GEOREF_CONFIG_STRING_TABLE{
    {alus::GeoTransformSourcePriority::PAM_INTERNAL_TABFILE_WORLDFILE_NONE, "PAM,INTERNAL,TABFILE,WORLDFILE,NONE"},
    {alus::GeoTransformSourcePriority::WORLDFILE_PAM_INTERNAL_TABFILE_NONE, "WORLDFILE,PAM,INTERNAL,TABFILE,NONE"}};
}

namespace alus {

// If one wants to install custom GDAL error handler -
// https://gdal.org/api/cpl.html#_CPPv418CPLSetErrorHandler15CPLErrorHandler

template <typename BufferType>
Dataset<BufferType>::Dataset(std::string_view filename) : file_path_(filename) {
    LoadDataset(filename);
}

template <typename BufferType>
Dataset<BufferType>::Dataset(GDALDataset* input_dataset) : dataset_{input_dataset} {
    LoadDataset("");
}

template <typename BufferType>
Dataset<BufferType>::Dataset(std::string_view filename, const GeoTransformSourcePriority& georef_source) : file_path_(filename) {
    CPLSetThreadLocalConfigOption("GDAL_GEOREF_SOURCES", GEOREF_CONFIG_STRING_TABLE.at(georef_source).c_str());
    try {
        LoadDataset(filename);
    } catch (...) {
        // Set back the defaults, since could not figure out how to apply "GEOREF_SOURCES" option to GDALOpen() call.
        CPLSetThreadLocalConfigOption(
            "GDAL_GEOREF_SOURCES",
            GEOREF_CONFIG_STRING_TABLE.at(GeoTransformSourcePriority::PAM_INTERNAL_TABFILE_WORLDFILE_NONE).c_str());
        throw;
    }
}

template <typename BufferType>
void Dataset<BufferType>::LoadDataset(std::string_view filename) {
    LoadDataset(filename, GA_ReadOnly);
}

template <typename BufferType>
std::tuple<double, double> Dataset<BufferType>::GetPixelCoordinatesFromIndex(int x, int y) const {
    auto const lon = x * this->pixel_size_lon_ + this->origin_lon_;  // Optional - {'+' (this->pixel_size_lon_ / 2)};
    auto const lat = y * this->pixel_size_lat_ + this->origin_lat_;  // Optional - {'+' (this->pixel_size_lat_ / 2)};
    return {lon, lat};
}

template <typename BufferType>
std::tuple<int, int> Dataset<BufferType>::GetPixelIndexFromCoordinates(double lon, double lat) const {
    auto const x = (lon - GetOriginLon()) / this->pixel_size_lon_;
    auto const y = (lat - GetOriginLat()) / this->pixel_size_lat_;

    return {x, y};
}

template <typename BufferType>
Dataset<BufferType>::~Dataset() {
    if (this->dataset_) {
        GDALClose(this->dataset_);
        this->dataset_ = nullptr;
    }
}

template <typename BufferType>
void Dataset<BufferType>::LoadRasterBand(int band_nr) {
    auto const bandCount = this->dataset_->GetRasterCount();
    if (bandCount == 0) {
        throw DatasetError("Does not support rasters with no bands.", this->dataset_->GetFileList()[0], 0);
    }

    if (bandCount < band_nr) {
        throw DatasetError("Too big band nr! You can not read a band that isn't there.",
                           this->dataset_->GetFileList()[0], 0);
    }
    this->x_size_ = this->dataset_->GetRasterBand(band_nr)->GetXSize();
    this->y_size_ = this->dataset_->GetRasterBand(band_nr)->GetYSize();
    this->data_buffer_.resize(this->x_size_ * this->y_size_);

    auto const inError = this->dataset_->GetRasterBand(band_nr)->RasterIO(GF_Read, 0, 0, this->x_size_, this->y_size_,
                                                                          this->data_buffer_.data(), this->x_size_,
                                                                          this->y_size_, gdal_data_type_, 0, 0);

    if (inError != CE_None) {
        throw DatasetError(CPLGetLastErrorMsg(), this->dataset_->GetFileList()[0], CPLGetLastErrorNo());
    }
}

template <typename BufferType>
Dataset<BufferType>::Dataset(GDALDataset& dataset) {
    this->dataset_ = &dataset;

    if (this->dataset_ == nullptr) {
        throw DatasetError(CPLGetLastErrorMsg(), this->dataset_->GetDescription(), CPLGetLastErrorNo());
    }

    if (this->dataset_->GetGeoTransform(this->transform_.data()) != CE_None) {
        throw DatasetError(CPLGetLastErrorMsg(), this->dataset_->GetFileList()[0], CPLGetLastErrorNo());
    }

    this->x_size_ = this->dataset_->GetRasterXSize();
    this->y_size_ = this->dataset_->GetRasterYSize();

    this->origin_lon_ = this->transform_[transform::TRANSFORM_LON_ORIGIN_INDEX];
    this->origin_lat_ = this->transform_[transform::TRANSFORM_LAT_ORIGIN_INDEX];
    this->pixel_size_lon_ = this->transform_[transform::TRANSFORM_PIXEL_X_SIZE_INDEX];
    this->pixel_size_lat_ = this->transform_[transform::TRANSFORM_PIXEL_Y_SIZE_INDEX];

    gdal_data_type_ = FindGdalDataType<BufferType>();
}
template <typename BufferType>
BufferType* Dataset<BufferType>::GetDeviceDataBuffer() {
    throw std::runtime_error("Get GPU buffer not implemented");
}

template <typename BufferType>
void Dataset<BufferType>::ReadRectangle(Rectangle rectangle, std::map<int, BufferType*>& bands){
    for (auto it = bands.begin(); it != bands.end(); ++it) {
        LOGV << it->first << ", " << it->second;
        ReadRectangle(rectangle, it->first, it->second);
    }
}

template <typename BufferType>
void Dataset<BufferType>::ReadRectangle(Rectangle rectangle, int band_nr, BufferType *data_buffer) {
    auto const bandCount = this->dataset_->GetRasterCount();
    if (bandCount == 0) {
        throw DatasetError("Does not support rasters with no bands.", this->dataset_->GetFileList()[0], 0);
    }

    if (bandCount < band_nr) {
        throw DatasetError("Too big band nr! You can not read a band that isn't there.",
                           this->dataset_->GetFileList()[0], 0);
    }
    if (rectangle.width == 0 || rectangle.height == 0) {
        throw DatasetError("Can not read a band with no numbers. ", this->dataset_->GetFileList()[0], 0);
    }

    auto const inError = this->dataset_->GetRasterBand(band_nr)->RasterIO(
        GF_Read, rectangle.x, rectangle.y, rectangle.width, rectangle.height, data_buffer, rectangle.width,
        rectangle.height, gdal_data_type_, 0, 0);

    if (inError != CE_None) {
        throw DatasetError(CPLGetLastErrorMsg(), this->dataset_->GetFileList()[0], CPLGetLastErrorNo());
    }
}
template <typename BufferType>
void Dataset<BufferType>::LoadDataset(std::string_view filename, GDALAccess access) {
    if (dataset_ == nullptr) {
        // TODO: move this to a place where it is unifiedly called once when system
        // starts.
        GDALAllRegister();  // Register all known drivers.
        this->dataset_ = (GDALDataset*)GDALOpen(filename.data(), GA_ReadOnly);
    }

    this->dataset_ = static_cast<GDALDataset*>(GDALOpen(filename.data(), access));
    if (this->dataset_ == nullptr) {
        throw DatasetError(CPLGetLastErrorMsg(), filename.data(), CPLGetLastErrorNo());
    }

    if (this->dataset_->GetGeoTransform(this->transform_.data()) == CE_None) {
        this->origin_lon_ = this->transform_[transform::TRANSFORM_LON_ORIGIN_INDEX];
        this->origin_lat_ = this->transform_[transform::TRANSFORM_LAT_ORIGIN_INDEX];
        this->pixel_size_lon_ = this->transform_[transform::TRANSFORM_PIXEL_X_SIZE_INDEX];
        this->pixel_size_lat_ = this->transform_[transform::TRANSFORM_PIXEL_Y_SIZE_INDEX];
    }

    gdal_data_type_ = FindGdalDataType<BufferType>();
}
template <typename BufferType>
Dataset<BufferType>::Dataset(std::string_view filename, GDALAccess access) : file_path_(filename) {
    LoadDataset(filename, access);
}
template <typename BufferType>
std::string_view Dataset<BufferType>::GetFilePath() {
    return file_path_;
}

template class Dataset<double>;
template class Dataset<float>;
template class Dataset<int16_t>;
template class Dataset<int>;
template class Dataset<Iq16>;
}  // namespace alus
