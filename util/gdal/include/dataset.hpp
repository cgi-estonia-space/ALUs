#pragma once

#include <array>
#include <stdexcept>
#include <string_view>
#include <tuple>
#include <vector>

#include <gdal_priv.h>

#include "raster_properties.hpp"

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

class Dataset {
   public:
    Dataset(std::string_view filename);
    void LoadRasterBand(int band_nr);
    Dataset(GDALDataset& dataset);

    GDALDataset* GetGdalDataset() { return dataset_; }

    Dataset(Dataset&& other) { *this = std::move(other); }

    Dataset& operator=(Dataset&& other) {
        this->dataset_ = other.dataset_;
        other.dataset_ = nullptr;
        this->origin_lat_ = other.origin_lat_;
        this->origin_lon_ = other.origin_lon_;
        this->pixel_size_lon_ = other.pixel_size_lon_;
        this->pixel_size_lat_ = other.pixel_size_lat_;
        this->x_size_ = other.x_size_;
        this->y_size_ = other.y_size_;
        this->data_buffer_ = std::move(other.data_buffer_);
        return *this;
    }

    Dataset(Dataset const&) = delete;
    Dataset& operator=(Dataset const&) = delete;

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
    std::vector<double> const& GetDataBuffer() const { return data_buffer_; }
    double GetNoDataValue() const { return no_data_value_; }

    ~Dataset();

   private:
    void LoadDataset(std::string_view filename);

    GDALDataset* dataset_;

    std::array<double, 6> transform_;
    double origin_lon_;
    double origin_lat_;
    double pixel_size_lon_;
    double pixel_size_lat_;

    double no_data_value_;

    int x_size_;
    int y_size_;
    std::vector<double> data_buffer_;

    // These are the TOP LEFT / UPPER LEFT coordinates of the image.
    static constexpr int TRANSFORM_LON_ORIGIN_INDEX{0};    // Or X origin
    static constexpr int TRANSFORM_LAT_ORIGIN_INDEX{3};    // Or Y origin
    static constexpr int TRANSFORM_PIXEL_X_SIZE_INDEX{1};  // Or pixel width
    static constexpr int TRANSFORM_PIXEL_Y_SIZE_INDEX{5};  // Or pixel height
};

}  // namespace alus
