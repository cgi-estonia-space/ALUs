#include "dataset.h"

#include <iostream>

#include <gdal.h>

namespace alus {

// If one wants to install custom GDAL error handler -
// https://gdal.org/api/cpl.html#_CPPv418CPLSetErrorHandler15CPLErrorHandler

Dataset::Dataset(std::string_view filename) { LoadDataset(filename); }

void Dataset::LoadDataset(std::string_view filename) {
    // TODO: move this to a place where it is unifiedly called once when system
    // starts.
    GDALAllRegister();  // Register all known drivers.

    this->dataset_ = (GDALDataset*)GDALOpen(filename.data(), GA_ReadOnly);
    if (this->dataset_ == nullptr) {
        throw DatasetError(CPLGetLastErrorMsg(), filename.data(),
                           CPLGetLastErrorNo());
    }

    if (this->dataset_->GetGeoTransform(this->transform_.data()) == CE_None) {
        this->origin_lon_ = this->transform_[TRANSFORM_LON_ORIGIN_INDEX];
        this->origin_lat_ = this->transform_[TRANSFORM_LAT_ORIGIN_INDEX];
        this->pixel_size_lon_ = this->transform_[TRANSFORM_PIXEL_X_SIZE_INDEX];
        this->pixel_size_lat_ = this->transform_[TRANSFORM_PIXEL_Y_SIZE_INDEX];
    }
}

std::tuple<double, double> Dataset::GetPixelCoordinatesFromIndex(int x,
                                                                 int y) const {
    auto const lon = x * this->pixel_size_lon_ +
                     this->origin_lon_;  // Optional - {'+' (this->pixel_size_lon_ / 2)};
    auto const lat = y * this->pixel_size_lat_ +
                     this->origin_lat_;  // Optional - {'+' (this->pixel_size_lat_ / 2)};
    return {lon, lat};
}

std::tuple<int, int> Dataset::GetPixelIndexFromCoordinates(double lon,
                                                           double lat) const {
    auto const x = (lon - GetOriginLon()) / this->pixel_size_lon_;
    auto const y = (lat - GetOriginLat()) / this->pixel_size_lat_;

    return {x, y};
}

Dataset::~Dataset() {
    if (this->dataset_) {
        GDALClose(this->dataset_);
        this->dataset_ = nullptr;
    }
}
void Dataset::LoadRasterBand(int band_nr) {
    auto const bandCount = this->dataset_->GetRasterCount();
    if (bandCount == 0) {
        throw DatasetError("Does not support rasters with no bands.",
                           this->dataset_->GetFileList()[0], 0);
    }

    if (bandCount < band_nr) {
        throw DatasetError("Too big band nr! You can not read a band that isn't there.",
                           this->dataset_->GetFileList()[0], 0);
    }
    this->x_size_ = this->dataset_->GetRasterXSize();
    this->y_size_ = this->dataset_->GetRasterYSize();
    this->data_buffer_.resize(this->x_size_ * this->y_size_);

    auto const inError = this->dataset_->GetRasterBand(band_nr)->RasterIO(
        GF_Read, 0, 0, this->x_size_, this->y_size_, this->data_buffer_.data(),
        this->x_size_, this->y_size_, GDALDataType::GDT_Float64, 0, 0);

    if (inError != CE_None) {
        throw DatasetError(CPLGetLastErrorMsg(), this->dataset_->GetFileList()[0],
                           CPLGetLastErrorNo());
    }

    this->no_data_value_ = this->dataset_->GetRasterBand(band_nr)->GetNoDataValue();
}
Dataset::Dataset(GDALDataset &dataset) {
    this->dataset_ = &dataset;

    if (this->dataset_ == nullptr) {
        throw DatasetError(CPLGetLastErrorMsg(), this->dataset_->GetDescription(),
                           CPLGetLastErrorNo());
    }

    if (this->dataset_->GetGeoTransform(this->transform_.data()) != CE_None) {
        throw DatasetError(CPLGetLastErrorMsg(), this->dataset_->GetFileList()[0],
                           CPLGetLastErrorNo());
    }

    this->x_size_ = this->dataset_->GetRasterXSize();
    this->y_size_ = this->dataset_->GetRasterYSize();

    this->origin_lon_ = this->transform_[TRANSFORM_LON_ORIGIN_INDEX];
    this->origin_lat_ = this->transform_[TRANSFORM_LAT_ORIGIN_INDEX];
    this->pixel_size_lon_ = this->transform_[TRANSFORM_PIXEL_X_SIZE_INDEX];
    this->pixel_size_lat_ = this->transform_[TRANSFORM_PIXEL_Y_SIZE_INDEX];
}

void Dataset::LoadRasterBandFloat(int band_nr) {
    auto const bandCount = this->dataset_->GetRasterCount();
    if (bandCount == 0) {
        throw DatasetError("Does not support rasters with no bands.",
                           this->dataset_->GetFileList()[0], 0);
    }

    if (bandCount < band_nr) {
        throw DatasetError("Too big band nr! You can not read a band that isn't there.",
                           this->dataset_->GetFileList()[0], 0);
    }
    this->x_size_ = this->dataset_->GetRasterXSize();
    this->y_size_ = this->dataset_->GetRasterYSize();
    this->float_data_buffer_.resize(this->x_size_ * this->y_size_);

    auto const inError = this->dataset_->GetRasterBand(band_nr)->RasterIO(
        GF_Read, 0, 0, this->x_size_, this->y_size_, this->float_data_buffer_.data(),
        this->x_size_, this->y_size_, GDALDataType::GDT_Float32, 0, 0);

    if (inError != CE_None) {
        throw DatasetError(CPLGetLastErrorMsg(), this->dataset_->GetFileList()[0],
                           CPLGetLastErrorNo());
    }
}
}  // namespace alus
