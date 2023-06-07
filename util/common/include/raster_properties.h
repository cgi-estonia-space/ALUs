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

#include <algorithm>
#include <array>

namespace alus {

struct RasterDimension final {
    int columnsX;
    int rowsY;

    RasterDimension& operator=(RasterDimension const& other) = default;
    RasterDimension(const RasterDimension& other) = default;

    RasterDimension operator+(int increase) const { return {this->columnsX + increase, this->rowsY + increase}; }

    void operator+=(int increase) { *this = *this + increase; }

    bool operator==(RasterDimension const& other) const {
        return this->columnsX == other.columnsX && this->rowsY == other.rowsY;
    }

    [[nodiscard]] size_t GetSize() const { return this->columnsX * this->rowsY; }
};

struct RasterPoint final {
    int const x;
    int const y;
};

using Coordinate = double;
using Longitude = Coordinate;
using Latitude = Coordinate;

struct Coordinates final {
    const Longitude lon;
    const Latitude lat;
};

struct PixelPosition final {
    int x;
    int y;
};

struct PrecisePixelPosition {
    double x;
    double y;
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
    static GeoTransformParameters BuildFromGdal(const double gdalGT[]) {
        return GeoTransformParameters{gdalGT[LON_ORIGIN_INDEX], gdalGT[LAT_ORIGIN_INDEX], gdalGT[PIXEL_X_SIZE_INDEX],
                                      gdalGT[PIXEL_Y_SIZE_INDEX]};
    }

    static std::array<double, GDAL_GEOTRANSFORM_PARAMETERS_LENGTH> ConvertToGdal(const GeoTransformParameters& params) {
        return {params.originLon, params.pixelSizeLon, 0.0, params.originLat, 0.0, params.pixelSizeLat};
    }

    // These are the TOP LEFT / UPPER LEFT coordinates of the image.
    static constexpr int LON_ORIGIN_INDEX{0};    // Or X origin
    static constexpr int LAT_ORIGIN_INDEX{3};    // Or Y origin
    static constexpr int PIXEL_X_SIZE_INDEX{1};  // Or pixel width
    static constexpr int PIXEL_Y_SIZE_INDEX{5};  // Or pixel height
};
}  // namespace alus