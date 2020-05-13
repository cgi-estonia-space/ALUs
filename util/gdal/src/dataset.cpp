#include "dataset.hpp"

#include <iostream>

#include <gdal.h>

namespace alus {

// If one wants to install custom GDAL error handler -
// https://gdal.org/api/cpl.html#_CPPv418CPLSetErrorHandler15CPLErrorHandler

Dataset::Dataset(std::string_view filename) { loadDataset(filename); }

void Dataset::loadDataset(std::string_view filename) {
    // TODO: move this to a place where it is unifiedly called once when system
    // starts.
    GDALAllRegister();  // Register all known drivers.

    this->dataset = (GDALDataset*)GDALOpen(filename.data(), GA_ReadOnly);
    if (this->dataset == nullptr) {
        throw DatasetError(CPLGetLastErrorMsg(), filename.data(),
                           CPLGetLastErrorNo());
    }

    if (this->dataset->GetGeoTransform(this->transform.data()) != CE_None) {
        throw DatasetError(CPLGetLastErrorMsg(), this->dataset->GetFileList()[0],
                           CPLGetLastErrorNo());
    }

    this->originLon = this->transform[TRANSFORM_LON_ORIGIN_INDEX];
    this->originLat = this->transform[TRANSFORM_LAT_ORIGIN_INDEX];
    this->pixelSizeLon = this->transform[TRANSFORM_PIXEL_X_SIZE_INDEX];
    this->pixelSizeLat = this->transform[TRANSFORM_PIXEL_Y_SIZE_INDEX];
}

std::tuple<double, double> Dataset::getPixelCoordinatesFromIndex(int x,
                                                                 int y) const {
    auto const lon = x * this->pixelSizeLon +
                     this->originLon;  // Optional - {'+' (this->pixelSizeLon / 2)};
    auto const lat = y * this->pixelSizeLat +
                     this->originLat;  // Optional - {'+' (this->pixelSizeLat / 2)};
    return {lon, lat};
}

std::tuple<int, int> Dataset::getPixelIndexFromCoordinates(double lon,
                                                           double lat) const {
    auto const x = (lon - getOriginLon()) / this->pixelSizeLon;
    auto const y = (lat - getOriginLat()) / this->pixelSizeLat;

    return {x, y};
}

Dataset::~Dataset() {
    if (this->dataset) {
        GDALClose(this->dataset);
        this->dataset = nullptr;
    }
}
void Dataset::loadRasterBand(int bandNr) {
    auto const bandCount = this->dataset->GetRasterCount();
    if (bandCount == 0) {
        throw DatasetError("Does not support rasters with no bands.",
                           this->dataset->GetFileList()[0], 0);
    }

    if (bandCount < bandNr) {
        throw DatasetError("Too big band nr! You can not read a band that isn't there.",
                           this->dataset->GetFileList()[0], 0);
    }
    this->xSize = this->dataset->GetRasterXSize();
    this->ySize = this->dataset->GetRasterYSize();
    this->dataBuffer.resize(this->xSize * this->ySize);

    auto const inError = this->dataset->GetRasterBand(bandNr)->RasterIO(
        GF_Read, 0, 0, this->xSize, this->ySize, this->dataBuffer.data(),
        this->xSize, this->ySize, GDALDataType::GDT_Float64, 0, 0);

    if (inError != CE_None) {
        throw DatasetError(CPLGetLastErrorMsg(), this->dataset->GetFileList()[0],
                           CPLGetLastErrorNo());
    }
}
}  // namespace alus
