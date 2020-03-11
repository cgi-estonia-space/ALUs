#include "dataset.hpp"

#include <iostream>

#include <gdal.h>

namespace slap {

// If one wants to install custom GDAL error handler - https://gdal.org/api/cpl.html#_CPPv418CPLSetErrorHandler15CPLErrorHandler

Dataset::Dataset(std::string_view filename) { loadDataset(filename); }

void Dataset::loadDataset(std::string_view filename) {
    GDALAllRegister();  // Register all known drivers

    m_dataset = (GDALDataset*)GDALOpen(filename.data(), GA_ReadOnly);
    if (m_dataset == nullptr) {
        throw DatasetError(
            CPLGetLastErrorMsg(),
            filename.data(), CPLGetLastErrorNo());
    }

    if (m_dataset->GetGeoTransform(m_transform.data()) != CE_None)
    {
        throw DatasetError(
            CPLGetLastErrorMsg(),
            m_dataset->GetFileList()[0], CPLGetLastErrorNo());
    }

    m_originLon = m_transform[TRANSFORM_LON_ORIGIN_INDEX];
    m_originLat = m_transform[TRANSFORM_LAT_ORIGIN_INDEX];
    m_pixelSizeLon = m_transform[TRANSFORM_PIXEL_X_SIZE_INDEX];
    m_pixelSizeLat = m_transform[TRANSFORM_PIXEL_Y_SIZE_INDEX];
}

std::tuple<double, double> Dataset::getPixelCoordinatesFromIndex(int x, int y) const {

    auto const lon = x * m_pixelSizeLon + m_originLon;// Optional - {'+' (m_pixelSizeLon / 2)};
    auto const lat = y * m_pixelSizeLat + m_originLat;// Optional - {'+' (m_pixelSizeLat / 2)};
    return {lon, lat};
}

std::tuple<int, int> Dataset::getPixelIndexFromCoordinates(double lon, double lat) const {

    auto const x = (lon - getOriginLon()) / m_pixelSizeLon;
    auto const y = (lat - getOriginLat()) / m_pixelSizeLat;

    return {x, y};
}

Dataset::~Dataset() {
    if (m_dataset) {
        GDALClose(m_dataset);
        m_dataset = nullptr;
    }
}
}  // namespace slap
