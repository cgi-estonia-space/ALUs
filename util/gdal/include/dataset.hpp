#pragma once

#include <array>
#include <stdexcept>
#include <string_view>
#include <tuple>
#include <vector>

#include <gdal_priv.h>

namespace slap {

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

    GDALDataset* getGDALDataset() { return m_dataset; }

    Dataset(Dataset&& other) {
        this->m_dataset = other.m_dataset;
        other.m_dataset = nullptr;
        this->m_originLat = other.m_originLat;
        this->m_originLon = other.m_originLon;
        this->m_pixelSizeLon = other.m_pixelSizeLon;
        this->m_pixelSizeLat = other.m_pixelSizeLat;
        this->m_band1Xsize = other.m_band1Xsize;
        this->m_band1Ysize = other.m_band1Ysize;
        this->m_band1Data = other.m_band1Data;
    }
    Dataset& operator=(Dataset&& other) {
        this->m_dataset = other.m_dataset;
        other.m_dataset = nullptr;
        this->m_originLat = other.m_originLat;
        this->m_originLon = other.m_originLon;
        this->m_pixelSizeLon = other.m_pixelSizeLon;
        this->m_pixelSizeLat = other.m_pixelSizeLat;
        return *this;
    }

    Dataset(Dataset const&) = delete;
    Dataset& operator=(Dataset const&) = delete;

    std::tuple<double /*lon*/, double /*lat*/> getPixelCoordinatesFromIndex(
        int x, int y) const;
    std::tuple<int /*x*/, int /*y*/> getPixelIndexFromCoordinates(
        double lon, double lat) const;

    double const* getTransform() const { return m_transform.data(); }

    /**
     * Origin is a TOP LEFT / UPPER LEFT corner of the image.
     * @return Longitude of the top left corner of the image.
     */
    double getOriginLon() const { return m_originLon; }
    /**
     * Origin is a TOP LEFT / UPPER LEFT corner of the image.
     * @return Latitude of the top left corner of the image.
     */
    double getOriginLat() const { return m_originLat; }

    int getBand1Xsize() const { return m_band1Xsize; }
    int getBand1Ysize() const { return m_band1Ysize; }
    std::vector<double> const& getBand1Data() const { return m_band1Data; }

    ~Dataset();

   private:
    void loadDataset(std::string_view filename);

    void loadRaster1Data();

    GDALDataset* m_dataset;

    std::array<double, 6> m_transform;
    double m_originLon;
    double m_originLat;
    double m_pixelSizeLon;
    double m_pixelSizeLat;

    int m_band1Xsize;
    int m_band1Ysize;
    std::vector<double> m_band1Data;

    // These are the TOP LEFT / UPPER LEFT coordinates of the image.
    static constexpr int TRANSFORM_LON_ORIGIN_INDEX{0};    // Or X origin
    static constexpr int TRANSFORM_LAT_ORIGIN_INDEX{3};    // Or Y origin
    static constexpr int TRANSFORM_PIXEL_X_SIZE_INDEX{1};  // Or pixel width
    static constexpr int TRANSFORM_PIXEL_Y_SIZE_INDEX{5};  // Or pixel height
};

}  // namespace slap
