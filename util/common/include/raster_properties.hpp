#pragma once

#include <algorithm>
#include <array>

namespace slap {

struct RasterDimension final {
    int columnsX;
    int rowsY;

    RasterDimension& operator= (RasterDimension const& other) = default;

    RasterDimension operator+ (int increase) const {
        return {this->columnsX + increase, this->rowsY + increase};
    }

    void operator+= (int increase) {
        *this = *this + increase;
    }

    bool operator== (RasterDimension const& other) const {
        return this->columnsX == other.columnsX && this->rowsY == other.rowsY;
    }

    [[nodiscard]] size_t getSize() const { return this->columnsX * this->rowsY; }
};

struct RasterPoint final {
    int const x;
    int const y;
};

struct Coordinates final {
    double const lon;
    double const lat;
};

struct GeoTransformParameters final {
    double originLon;
    double originLat;
    double pixelSizeLon;
    double pixelSizeLat;
};

class GeoTransformConstruct final {
   public:
    static constexpr int GDAL_GEOTRANSFORM_PARAMETERS_LENGTH{6};
    static GeoTransformParameters buildFromGDAL(double gdalGT[]) {
        return GeoTransformParameters{
            gdalGT[LON_ORIGIN_INDEX], gdalGT[LAT_ORIGIN_INDEX], gdalGT[PIXEL_X_SIZE_INDEX], gdalGT[PIXEL_Y_SIZE_INDEX]};
    }

    // These are the TOP LEFT / UPPER LEFT coordinates of the image.
    static constexpr int LON_ORIGIN_INDEX{0};    // Or X origin
    static constexpr int LAT_ORIGIN_INDEX{3};    // Or Y origin
    static constexpr int PIXEL_X_SIZE_INDEX{1};  // Or pixel width
    static constexpr int PIXEL_Y_SIZE_INDEX{5};  // Or pixel height
};
}  // namespace slap