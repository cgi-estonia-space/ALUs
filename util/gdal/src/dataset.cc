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
#include <mutex>
#include <thread>

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
Dataset<BufferType>::Dataset(std::string_view filename, const GeoTransformSourcePriority& georef_source)
    : file_path_(filename) {
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
    if (cacher_.joinable()) {
        is_allowed_to_cache_ = false;
        cacher_.join();
    }

    if (this->dataset_) {
        GDALClose(this->dataset_);
        this->dataset_ = nullptr;
    }
}

template <typename BufferType>
void Dataset<BufferType>::LoadRasterBand(int band_nr) {
    auto const band_count = this->dataset_->GetRasterCount();
    if (band_count == 0) {
        throw DatasetError("Does not support rasters with no bands.");
    }

    if (band_count < band_nr) {
        throw DatasetError("Too big band nr! You can not read a band that isn't there.");
    }

    reading_area_ = {0, 0, this->dataset_->GetRasterBand(band_nr)->GetXSize(),
                     this->dataset_->GetRasterBand(band_nr)->GetYSize()};
    this->data_buffer_.resize(reading_area_.width * reading_area_.height);

    auto const in_error = this->dataset_->GetRasterBand(band_nr)->RasterIO(
        GF_Read, 0, 0, reading_area_.width, reading_area_.height, this->data_buffer_.data(), reading_area_.width,
        reading_area_.height, gdal_data_type_, 0, 0);

    if (in_error != CE_None) {
        throw DatasetError(CPLGetLastErrorMsg());
    }
}

template <typename BufferType>
GDALDataset* Dataset<BufferType>::GetGdalDataset() {
    if (cacher_.joinable()) {
        is_allowed_to_cache_ = false;
        cacher_.join();

        if (cacher_exception_ != nullptr) {
            LOGE << "Rethrowing cacher exception" << std::endl;
            std::rethrow_exception(cacher_exception_);
        }
    }

    return dataset_;
}

template <typename BufferType>
Dataset<BufferType>::Dataset(GDALDataset& dataset) {
    this->dataset_ = &dataset;

    if (this->dataset_ == nullptr) {
        throw DatasetError(CPLGetLastErrorMsg());
    }

    if (this->dataset_->GetGeoTransform(this->transform_.data()) != CE_None) {
        throw DatasetError(CPLGetLastErrorMsg());
    }

    reading_area_ = {0, 0, this->dataset_->GetRasterXSize(), this->dataset_->GetRasterYSize()};

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
void Dataset<BufferType>::ReadRectangle(Rectangle rectangle, std::map<int, BufferType*>& bands) {
    for (auto it = bands.begin(); it != bands.end(); ++it) {
        LOGV << it->first << ", " << it->second;
        ReadRectangle(rectangle, it->first, it->second, false, reading_area_.x, reading_area_.y);
    }
}

template <typename BufferType>
void Dataset<BufferType>::ReadRectangle(Rectangle rectangle, int band_nr, BufferType* data_buffer) {
    ReadRectangle(rectangle, band_nr, data_buffer, false, reading_area_.x, reading_area_.y);
}

template <typename BufferType>
void Dataset<BufferType>::ReadRectangle(Rectangle rectangle, int band_nr, BufferType* data_buffer, bool is_from_cache,
                                        int offset_x, int offset_y) {
    auto const band_count = this->dataset_->GetRasterCount();
    if (band_count == 0) {
        throw DatasetError("Does not support rasters with no bands.");
    }

    if (band_count < band_nr) {
        throw DatasetError("Too big band nr! You can not read a band that isn't there.");
    }
    if (rectangle.width == 0 || rectangle.height == 0) {
        throw DatasetError("Can not read a band with no numbers. ");
    }

    if (!is_from_cache) {
        is_allowed_to_cache_ = false;
    }

    if (rectangle.x + offset_x + rectangle.width > dataset_->GetRasterXSize() ||
        rectangle.y + offset_y + rectangle.height > dataset_->GetRasterYSize()) {
        throw DatasetError("Rectangle out of image bounds");
    }

    if (!is_from_cache && (rectangle.x + rectangle.width > reading_area_.width ||
                           rectangle.y + rectangle.height > reading_area_.height)) {
        throw DatasetError("Rectangle out of reading area bounds");
    }

    std::unique_lock lock(read_lock_);
    if (cacher_exception_ != nullptr) {
        LOGE << "Rethrowing cacher exception" << std::endl;
        std::rethrow_exception(cacher_exception_);
    }

    auto const in_error = this->dataset_->GetRasterBand(band_nr)->RasterIO(
        GF_Read, rectangle.x + offset_x, rectangle.y + offset_y, rectangle.width, rectangle.height, data_buffer,
        rectangle.width, rectangle.height, gdal_data_type_, 0, 0);

    if (in_error != CE_None) {
        throw DatasetError(CPLGetLastErrorMsg());
    }
}

template <typename BufferType>
void Dataset<BufferType>::LoadDataset(std::string_view filename, GDALAccess access) {

    this->dataset_ = static_cast<GDALDataset*>(GDALOpen(filename.data(), access));
    if (this->dataset_ == nullptr) {
        throw DatasetError(CPLGetLastErrorMsg());
    }

    if (this->dataset_->GetGeoTransform(this->transform_.data()) == CE_None) {
        this->origin_lon_ = this->transform_[transform::TRANSFORM_LON_ORIGIN_INDEX];
        this->origin_lat_ = this->transform_[transform::TRANSFORM_LAT_ORIGIN_INDEX];
        this->pixel_size_lon_ = this->transform_[transform::TRANSFORM_PIXEL_X_SIZE_INDEX];
        this->pixel_size_lat_ = this->transform_[transform::TRANSFORM_PIXEL_Y_SIZE_INDEX];
    }

    gdal_data_type_ = FindGdalDataType<BufferType>();
    reading_area_ = {0, 0, dataset_->GetRasterXSize(), dataset_->GetRasterYSize()};
}
template <typename BufferType>
Dataset<BufferType>::Dataset(std::string_view filename, GDALAccess access) : file_path_(filename) {
    LoadDataset(filename, access);
}
template <typename BufferType>
std::string_view Dataset<BufferType>::GetFilePath() const {
    return file_path_;
}

template <typename BufferType>
void Dataset<BufferType>::CacheImage() {
    const int default_y = 100;
    int offset_y;
    int actual_y;

    try {
        const int band_count = this->dataset_->GetRasterCount();
        std::vector<BufferType> temp_pile(default_y);
        const int last_y = reading_area_.y + reading_area_.height;

        int cached_blocks = 0;

        for (int i = 1; i <= band_count; i++) {
            for (offset_y = reading_area_.y; offset_y < last_y; offset_y += default_y) {
                if ((offset_y + default_y) >= last_y) {
                    actual_y = last_y - offset_y;
                } else {
                    actual_y = default_y;
                }
                ReadRectangle({0, offset_y, 1, actual_y}, i, temp_pile.data(), true, 0, 0);
                cached_blocks++;
                if (!is_allowed_to_cache_) {
                    break;
                }
            }
        }

        const int total_blocks = (reading_area_.height + default_y - 1) / default_y;
        const double percent = (100.0 * cached_blocks) / (band_count * total_blocks);
        LOGD << "Dataset pre-cached " << percent << "% of file " << file_path_;
    } catch (const std::exception&) {
        cacher_exception_ = std::current_exception();
    }
}

template <typename BufferType>
void Dataset<BufferType>::TryToCacheImage() {
    if (!cacher_.joinable()) {
        is_allowed_to_cache_ = true;
        cacher_ = std::thread(&Dataset<BufferType>::CacheImage, this);
    }
}

template class Dataset<double>;
template class Dataset<float>;
template class Dataset<int16_t>;
template class Dataset<uint16_t>;
template class Dataset<int>;
template class Dataset<Iq16>;
}  // namespace alus
