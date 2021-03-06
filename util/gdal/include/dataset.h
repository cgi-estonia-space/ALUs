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
#pragma once

#include <array>
#include <atomic>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string_view>
#include <thread>
#include <tuple>
#include <vector>

#include <gdal_priv.h>

#include "alus_file_reader.h"
#include "raster_properties.h"
#include "transform_constants.h"

namespace alus {

class DatasetError : public std::runtime_error {
public:
    explicit DatasetError(std::string_view what) : std::runtime_error(what.data()) {}
};

/**
 * This enum enables to force georeferencing source for GDAL driver as specified here:
 * https://gdal.org/drivers/raster/gtiff.html#georeferencing
 */
enum class GeoTransformSourcePriority {
    PAM_INTERNAL_TABFILE_WORLDFILE_NONE,  // This is a default for GDAL driver, essentially no need to specify this.
    WORLDFILE_PAM_INTERNAL_TABFILE_NONE
};

template <typename BufferType>
class Dataset : public AlusFileReader<BufferType> {
public:
    Dataset() = default;
    explicit Dataset(std::string_view filename);
    explicit Dataset(std::string_view filename, GDALAccess access);
    explicit Dataset(GDALDataset* input_dataset);
    explicit Dataset(std::string_view filename, const GeoTransformSourcePriority& georef_source);
    void LoadRasterBand(int band_nr) override;
    explicit Dataset(GDALDataset& dataset);

    GDALDataset* GetGdalDataset() { return dataset_; }

    Dataset(Dataset<BufferType>&& other) noexcept { *this = std::move(other); }

    Dataset<BufferType>& operator=(Dataset<BufferType>&& other) noexcept {
        this->dataset_ = other.dataset_;
        other.dataset_ = nullptr;
        this->origin_lat_ = other.origin_lat_;
        this->origin_lon_ = other.origin_lon_;
        this->pixel_size_lon_ = other.pixel_size_lon_;
        this->pixel_size_lat_ = other.pixel_size_lat_;
        this->gdal_data_type_ = other.gdal_data_type_;
        this->reading_area_ = other.reading_area_;
        this->data_buffer_ = std::move(other.data_buffer_);
        return *this;
    }

    Dataset(Dataset<BufferType> const&) = delete;
    Dataset<BufferType>& operator=(Dataset<BufferType> const&) = delete;

    [[nodiscard]] std::tuple<double /*lon*/, double /*lat*/> GetPixelCoordinatesFromIndex(int x, int y) const;
    [[nodiscard]] std::tuple<int /*x*/, int /*y*/> GetPixelIndexFromCoordinates(double lon, double lat) const;

    [[nodiscard]] double const* GetTransform() const { return transform_.data(); }

    /**
     * Origin is a TOP LEFT / UPPER LEFT corner of the image.
     * @return Longitude of the top left corner of the image.
     */
    [[nodiscard]] double GetOriginLon() const { return origin_lon_; }
    /**
     * Origin is a TOP LEFT / UPPER LEFT corner of the image.
     * @return Latitude of the top left corner of the image.
     */
    [[nodiscard]] double GetOriginLat() const { return origin_lat_; }

    [[nodiscard]] double GetPixelSizeLon() const { return pixel_size_lon_; }
    [[nodiscard]] double GetPixelSizeLat() const { return pixel_size_lat_; }

    void FillGeoTransform(double& origin_lon, double& origin_lat, double& pixel_size_lon,
                          double& pixel_size_lat) const {
        origin_lon = GetOriginLon();
        origin_lat = GetOriginLat();
        pixel_size_lon = GetPixelSizeLon();
        pixel_size_lat = GetPixelSizeLat();
    }

    [[nodiscard]] int GetRasterSizeX() const { return reading_area_.width; }
    [[nodiscard]] int GetRasterSizeY() const { return reading_area_.height; }
    [[nodiscard]] RasterDimension GetRasterDimensions() const { return {GetRasterSizeX(), GetRasterSizeY()}; }
    std::vector<BufferType> const& GetHostDataBuffer() const override { return data_buffer_; }
    [[nodiscard]] size_t GetBandCount() const { return dataset_->GetBands().size(); }
    double GetNoDataValue(int band_nr) { return dataset_->GetRasterBand(band_nr)->GetNoDataValue(); }
    uint64_t GetBufferByteSize() override { return data_buffer_.size() * sizeof(BufferType); };
    size_t GetBufferElemCount() override { return data_buffer_.size(); };
    BufferType* GetDeviceDataBuffer() override;
    void ReadRectangle(Rectangle rectangle, int band_nr, BufferType* data_buffer) override;
    [[nodiscard]] std::string_view GetFilePath();
    void ReadRectangle(Rectangle rectangle, std::map<int, BufferType*>& bands) override;
    void TryToCacheImage() override;
    void SetReadingArea(Rectangle new_area) override { reading_area_ = new_area; };

    virtual ~Dataset();

private:
    void LoadDataset(std::string_view filename);
    void LoadDataset(std::string_view filename, GDALAccess access);
    void CacheImage();
    void ReadRectangle(Rectangle rectangle, int band_nr, BufferType* data_buffer, bool is_from_cache, int offset_x,
                       int offset_y);
    std::mutex read_lock_;  // rasterIO is not actually thread safe. Some stream object is shared.
    std::thread cacher_;
    Rectangle reading_area_;

    GDALDataset* dataset_{};
    GDALDataType gdal_data_type_;

    std::array<double, 6> transform_{};  // NOLINT
    double origin_lon_{};
    double origin_lat_{};
    double pixel_size_lon_{};
    double pixel_size_lat_{};
    std::atomic<bool> is_allowed_to_cache_{true};

    std::vector<BufferType> data_buffer_{};
    std::string file_path_{};

    std::exception_ptr cacher_exception_{nullptr};
};

}  // namespace alus
