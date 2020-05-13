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
    DatasetError(std::string what, std::string filename, int errorCode)
        : std::runtime_error(what),
          m_what{std::move(what)},
          m_fileName{std::move(filename)},
          m_errorCode{errorCode} {}

   private:
    std::string m_what;
    std::string m_fileName;
    int m_errorCode;
};

class Dataset {
   public:
    Dataset(std::string_view filename);
    void loadRasterBand(int bandNr);

    GDALDataset* getGDALDataset() { return dataset; }

    Dataset(Dataset&& other) { *this = std::move(other); }

    Dataset& operator=(Dataset&& other) {
        this->dataset = other.dataset;
        other.dataset = nullptr;
        this->originLat = other.originLat;
        this->originLon = other.originLon;
        this->pixelSizeLon = other.pixelSizeLon;
        this->pixelSizeLat = other.pixelSizeLat;
        this->xSize = other.xSize;
        this->ySize = other.ySize;
        this->dataBuffer = std::move(other.dataBuffer);
        return *this;
    }

    Dataset(Dataset const&) = delete;
    Dataset& operator=(Dataset const&) = delete;

    std::tuple<double /*lon*/, double /*lat*/> getPixelCoordinatesFromIndex(
        int x, int y) const;
    std::tuple<int /*x*/, int /*y*/> getPixelIndexFromCoordinates(
        double lon, double lat) const;

    double const* getTransform() const { return transform.data(); }

    /**
     * Origin is a TOP LEFT / UPPER LEFT corner of the image.
     * @return Longitude of the top left corner of the image.
     */
    double getOriginLon() const { return originLon; }
    /**
     * Origin is a TOP LEFT / UPPER LEFT corner of the image.
     * @return Latitude of the top left corner of the image.
     */
    double getOriginLat() const { return originLat; }

    double getPixelSizeLon() const { return pixelSizeLon; }
    double getPixelSizeLat() const { return pixelSizeLat; }

    void fillGeoTransform(double& originLon, double& originLat,
                          double& pixelSizeLon, double& pixelSizeLat) const {
        originLon = getOriginLon();
        originLat = getOriginLat();
        pixelSizeLon = getPixelSizeLon();
        pixelSizeLat = getPixelSizeLat();
    }

    int getRasterSizeX() const { return dataset->GetRasterXSize(); }
    int getRasterSizeY() const { return dataset->GetRasterYSize(); }
    RasterDimension getRasterDimensions() const { return {getRasterSizeX(), getRasterSizeY()}; }
    int getColumnCount() const { return getRasterSizeX(); }
    int getRowCount() const { return getRasterSizeY(); }
    int getXSize() const { return xSize; }
    int getYSize() const { return ySize; }
    std::vector<double> const& getDataBuffer() const { return dataBuffer; }

    ~Dataset();

   private:
    void loadDataset(std::string_view filename);

    GDALDataset* dataset;

    std::array<double, 6> transform;
    double originLon;
    double originLat;
    double pixelSizeLon;
    double pixelSizeLat;

    int xSize;
    int ySize;
    std::vector<double> dataBuffer;

    // These are the TOP LEFT / UPPER LEFT coordinates of the image.
    static constexpr int TRANSFORM_LON_ORIGIN_INDEX{0};    // Or X origin
    static constexpr int TRANSFORM_LAT_ORIGIN_INDEX{3};    // Or Y origin
    static constexpr int TRANSFORM_PIXEL_X_SIZE_INDEX{1};  // Or pixel width
    static constexpr int TRANSFORM_PIXEL_Y_SIZE_INDEX{5};  // Or pixel height
};

}  // namespace alus
