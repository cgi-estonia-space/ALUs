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
#include <stdexcept>
#include <string_view>
#include <tuple>
#include <vector>

#include <gdal_priv.h>

#include "raster_properties.hpp"
#include "alus_file_reader.h"
#include "transform_constants.h"

namespace alus {

class DatasetError : public std::runtime_error {
   public:
    DatasetError(std::string what, std::string filename, int error_code)
        : std::runtime_error(what), m_what{std::move(what)}, m_fileName{std::move(filename)}, m_errorCode{error_code} {}

   private:
    std::string m_what;
    std::string m_fileName;
    int m_errorCode;
};


/**
 * This enum enables to force georeferencing source for GDAL driver as specified here:
 * https://gdal.org/drivers/raster/gtiff.html#georeferencing
 */
enum class GeoTransformSourcePriority {
    PAM_INTERNAL_TABFILE_WORLDFILE_NONE, // This is a default for GDAL driver, essentially no need to specify this.
    WORLDFILE_PAM_INTERNAL_TABFILE_NONE
};

template <typename BufferType>
class Dataset: public AlusFileReader<BufferType> {
   public:



    Dataset() = default;
    explicit Dataset(std::string_view filename);
    explicit Dataset(GDALDataset* input_dataset);
    explicit Dataset(std::string_view filename, const GeoTransformSourcePriority& georef_source);
    void LoadRasterBand(int band_nr) override ;
    Dataset(GDALDataset& dataset);


    GDALDataset* GetGdalDataset() { return dataset_; }

    Dataset(Dataset<BufferType>&& other) { *this = std::move(other); }

    Dataset<BufferType>& operator=(Dataset<BufferType>&& other) {
        this->dataset_ = other.dataset_;
        other.dataset_ = nullptr;
        this->origin_lat_ = other.origin_lat_;
        this->origin_lon_ = other.origin_lon_;
        this->pixel_size_lon_ = other.pixel_size_lon_;
        this->pixel_size_lat_ = other.pixel_size_lat_;
        this->x_size_ = other.x_size_;
        this->y_size_ = other.y_size_;
        this->gdal_data_type_ = other.gdal_data_type_;
        this->data_buffer_ = std::move(other.data_buffer_);
        return *this;
    }

    Dataset(Dataset<BufferType> const&) = delete;
    Dataset<BufferType>& operator=(Dataset<BufferType> const&) = delete;

    std::tuple<double /*lon*/, double /*lat*/> GetPixelCoordinatesFromIndex(int x, int y) const;
    std::tuple<int /*x*/, int /*y*/> GetPixelIndexFromCoordinates(double lon, double lat) const;

    double const* GetTransform() const { return transform_.data(); }

    /**
     * Origin is a TOP LEFT / UPPER LEFT corner of the image.
     * @return Longitude of the top left corner of the image.
     */
    double GetOriginLon() const { return origin_lon_; }
    /**
     * Origin is a TOP LEFT / UPPER LEFT corner of the image.
     * @return Latitude of the top left corner of the image.
     */
    double GetOriginLat() const { return origin_lat_; }

    double GetPixelSizeLon() const { return pixel_size_lon_; }
    double GetPixelSizeLat() const { return pixel_size_lat_; }

    void FillGeoTransform(double& origin_lon,
                          double& origin_lat,
                          double& pixel_size_lon,
                          double& pixel_size_lat) const {
        origin_lon = GetOriginLon();
        origin_lat = GetOriginLat();
        pixel_size_lon = GetPixelSizeLon();
        pixel_size_lat = GetPixelSizeLat();
    }

    int GetRasterSizeX() const { return dataset_->GetRasterXSize(); }
    int GetRasterSizeY() const { return dataset_->GetRasterYSize(); }
    RasterDimension GetRasterDimensions() const { return {GetRasterSizeX(), GetRasterSizeY()}; }
    int GetColumnCount() const { return GetRasterSizeX(); }
    int GetRowCount() const { return GetRasterSizeY(); }
    int GetXSize() const { return x_size_; }
    int GetYSize() const { return y_size_; }
    std::vector<BufferType> const& GetHostDataBuffer() const override { return data_buffer_; }
    BufferType GetNoDataValue(int band_nr) { return dataset_->GetRasterBand(band_nr)->GetNoDataValue(); }
    long unsigned int GetBufferByteSize() override { return data_buffer_.size() * sizeof(BufferType);};
    size_t GetBufferElemCount() override { return data_buffer_.size();};
    BufferType* GetDeviceDataBuffer() override ;
    void ReadRectangle(Rectangle rectangle, int band_nr, BufferType*data_buffer) override ;

    ~Dataset();

   private:
    void LoadDataset(std::string_view filename);

    GDALDataset* dataset_{};
    GDALDataType gdal_data_type_;

    std::array<double, 6> transform_{};
    double origin_lon_{};
    double origin_lat_{};
    double pixel_size_lon_{};
    double pixel_size_lat_{};

    int x_size_{};
    int y_size_{};
    std::vector<BufferType> data_buffer_{};

};

}  // namespace alus
