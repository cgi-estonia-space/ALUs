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

#include <cstdint>
#include <string>
#include <string_view>
#include <typeindex>
#include <typeinfo>

#include <cpl_error.h>
#include <gdal.h>
#include <gdal_priv.h>

#include "metadata_record.h"
#include "target_dataset.h"
#include "type_parameter.h"

#define CHECK_GDAL_ERROR(err) alus::CheckGdalError(err, __FILE__, __LINE__)
#define CHECK_GDAL_PTR(ptr) CHECK_GDAL_ERROR((ptr) == nullptr ? CE_Failure : CE_None)
#define CHECK_OGR_ERROR(err) alus::CheckOgrError(err, __FILE__, __LINE__)

namespace alus {
namespace gdal::constants {
// GDAL driver constants
constexpr char GDAL_MEM_DRIVER[]{"MEM"};
constexpr char GDAL_GTIFF_DRIVER[]{"GTiff"};
constexpr char SQLITE_DRIVER[]{"SQLite"};
constexpr char GDAL_GTIFF_FILE_EXTENSION[]{".tif"};
constexpr int GDAL_DEFAULT_RASTER_BAND{1};
constexpr int GDAL_GEOTRANSFORM_PARAMETER_COUNT{6};

constexpr std::string_view GDAL_ZIP_PREFIX{"/vsizip/"};
constexpr std::string_view GDAL_GZIP_PREFIX{"/vsigzip/"};
constexpr std::string_view GDAL_TAR_PREFIX{"/vsitar/"};

constexpr std::string_view ZIP_EXTENSION{".zip"};
constexpr std::string_view GZIP_EXTENSION{".gz"};
constexpr std::string_view TAR_EXTENSION{".tar"};
constexpr std::string_view TGZ_EXTENSION{
    ".tgz"};  // TODO(Anton): .tar.gz is not supported yet (would require additional checks in AdjustFilePath()

constexpr std::string_view SUBDATASET_KEY{"SUBDATASETS"};
}  // namespace gdal::constants

class GdalErrorException final : public std::runtime_error {
public:
    GdalErrorException(CPLErr const errType, CPLErrorNum const errNum, std::string_view errMsg, std::string_view src,
                       int srcLine)
        : std::runtime_error("GDAL error no " + std::to_string(static_cast<int>(errNum)) + " type " +
                             std::to_string(static_cast<int>(errType)) + " - '" + std::string{errMsg} + "' at " +
                             std::string{src} + ":" + std::to_string(srcLine)),
          gdal_error_{errNum},
          file_{src},
          line_{srcLine} {}

    [[nodiscard]] CPLErrorNum GetGdalError() const { return gdal_error_; }
    [[nodiscard]] std::string_view GetSource() const { return file_; }
    [[nodiscard]] int GetLine() const { return line_; }

private:
    CPLErrorNum const gdal_error_;
    std::string file_;
    int const line_;
};

class OgrErrorException final : public std::runtime_error {
public:
    OgrErrorException(OGRErr const err_num, std::string_view src, int srcLine)
        : std::runtime_error("OGR error no '" + std::to_string(static_cast<int>(err_num)) + "' at " + std::string{src} +
                             ":" + std::to_string(srcLine)),
          error_{err_num},
          file_{src},
          line_{srcLine} {}

    [[nodiscard]] OGRErr GetOgrError() const { return error_; }
    [[nodiscard]] std::string_view GetSource() const { return file_; }
    [[nodiscard]] int GetLine() const { return line_; }

private:
    OGRErr const error_;
    std::string file_;
    int const line_;
};

inline void CheckGdalError(CPLErr const err, char const* file, int const line) {
    if (err != CE_None) {
        throw alus::GdalErrorException(err, CPLGetLastErrorNo(), CPLGetLastErrorMsg(), file, line);
    }
}

inline void CheckOgrError(OGRErr err, char const* file, int const line) {
    if (err != OGRERR_NONE) {
        throw alus::OgrErrorException(err, file, line);
    }
}

inline GDALDriver* GetGdalSqliteDriver() {
    return GetGDALDriverManager()->GetDriverByName(alus::gdal::constants::SQLITE_DRIVER);
}

inline GDALDriver* GetGdalMemDriver() {
    return GetGDALDriverManager()->GetDriverByName(alus::gdal::constants::GDAL_MEM_DRIVER);
}

inline GDALDriver* GetGdalGeoTiffDriver() {
    return GetGDALDriverManager()->GetDriverByName(alus::gdal::constants::GDAL_GTIFF_DRIVER);
}

struct Iq16 {
    int16_t i;
    int16_t q;
};
static_assert(sizeof(Iq16) == 4, "Do no alter the memory layout of this structure!");  // NOLINT

template <typename BufferType>
GDALDataType FindGdalDataType() {
    if (std::is_same_v<BufferType, double>) {
        return GDALDataType::GDT_Float64;
    }
    if (std::is_same_v<BufferType, float>) {
        return GDALDataType::GDT_Float32;
    }
    if (std::is_same_v<BufferType, int16_t>) {
        return GDALDataType::GDT_Int16;
    }
    if (std::is_same_v<BufferType, uint16_t>) {
        return GDALDataType::GDT_UInt16;
    }
    if (std::is_same_v<BufferType, int32_t>) {
        return GDALDataType::GDT_Int32;
    }
    if (std::is_same_v<BufferType, Iq16>) {
        return GDALDataType::GDT_CInt16;
    }
    // todo this function and error can be compile time, but requires refactoring in other places
    throw std::invalid_argument(std::string(typeid(BufferType).name()) +
                                " is not an implemented type for this dataset.");
}

inline TypeParameters CreateTypeParametersFrom(GDALDataType dt) {
    switch (dt) {
        case GDALDataType::GDT_Byte:
            return TypeParameters::CreateFor<uint8_t>();
        case GDALDataType::GDT_UInt16:
            return TypeParameters::CreateFor<uint16_t>();
        case GDALDataType::GDT_Int16:
            return TypeParameters::CreateFor<int16_t>();
        case GDALDataType::GDT_UInt32:
            return TypeParameters::CreateFor<uint32_t>();
        case GDALDataType::GDT_Int32:
            return TypeParameters::CreateFor<int32_t>();
        case GDALDataType::GDT_Float32:
            return TypeParameters::CreateFor<float>();
        case GDALDataType::GDT_Float64:
            return TypeParameters::CreateFor<double>();
        default:
            throw std::invalid_argument("Unsupported GDAL data type - " + std::to_string(dt));
    }
}

void GeoTiffWriteFile(GDALDataset* input_dataset, std::string_view output_file);

void AddMetadataTo(GDALDataset* ds, const common::metadata::Container& md);

template <typename T>
void WriteSimpleDatasetToGeoTiff(std::vector<SimpleDataset<T>>& ds, std::string_view path,
                                 const std::vector<std::pair<std::string, std::string>>& options,
                                 const common::metadata::Container& md, bool release_band_buffer = false) {
    auto output_driver = GetGdalGeoTiffDriver();
    CHECK_GDAL_PTR(output_driver);

    const auto gdal_dt = FindGdalDataType<T>();
    char** output_driver_options = nullptr;
    for (const auto& option : options) {
        output_driver_options = CSLSetNameValue(output_driver_options, option.first.c_str(), option.second.c_str());
    }

    auto csl_destroy = [](char** options) { CSLDestroy(options); };
    std::unique_ptr<char*, decltype(csl_destroy)> driver_opt_guard(output_driver_options, csl_destroy);
    auto gdal_ds = output_driver->Create(path.data(), ds.front().width, ds.front().height, ds.size(), gdal_dt,
                                         output_driver_options);
    CHECK_GDAL_PTR(gdal_ds);
    std::unique_ptr<GDALDataset, decltype(&GDALClose)> gdal_close_guard(gdal_ds, GDALClose);
    CHECK_GDAL_ERROR(gdal_ds->SetGeoTransform(ds.front().geo_transform));
    CHECK_GDAL_ERROR(gdal_ds->SetProjection(ds.front().projection_wkt.data()));
    auto band_index{gdal::constants::GDAL_DEFAULT_RASTER_BAND};
    for (auto& band : ds) {
        CHECK_GDAL_ERROR(gdal_ds->GetRasterBand(band_index)->SetNoDataValue(band.no_data));
        CHECK_GDAL_ERROR(gdal_ds->GetRasterBand(band_index)
                             ->RasterIO(GF_Write, 0, 0, band.width, band.height, band.buffer.get(), band.width,
                                        band.height, gdal_dt, 0, 0));
        if (release_band_buffer) {
            band.buffer.reset();
        }
        band_index++;
    }
    AddMetadataTo(gdal_ds, md);
}

template <typename T>
void FetchSimpleDatasetFromGdalDataset(SimpleDataset<T>& ds, GDALDataset* gdal_ds) {
    const auto width = gdal_ds->GetRasterBand(gdal::constants::GDAL_DEFAULT_RASTER_BAND)->GetXSize();
    const auto height = gdal_ds->GetRasterBand(gdal::constants::GDAL_DEFAULT_RASTER_BAND)->GetYSize();

    ds.buffer = std::unique_ptr<T[]>(new T[width * height]);
    ds.width = width;
    ds.height = height;
    ds.no_data = static_cast<T>(gdal_ds->GetRasterBand(gdal::constants::GDAL_DEFAULT_RASTER_BAND)->GetNoDataValue());
    CHECK_GDAL_ERROR(gdal_ds->GetGeoTransform(ds.geo_transform));
    ds.projection_wkt = gdal_ds->GetProjectionRef();

    CHECK_GDAL_ERROR(
        gdal_ds->GetRasterBand(gdal::constants::GDAL_DEFAULT_RASTER_BAND)
            ->RasterIO(GF_Read, 0, 0, width, height, ds.buffer.get(), width, height, FindGdalDataType<T>(), 0, 0));
}

template <typename T>
void StoreSimpleDatasetToGdalRasterBand(SimpleDataset<T>& ds, GDALRasterBand* gdal_band) {
    CHECK_GDAL_ERROR(gdal_band->RasterIO(GF_Write, 0, 0, ds.width, ds.height, ds.buffer.get(), ds.width, ds.height,
                                         FindGdalDataType<T>(), 0, 0));
}

std::string FindOptimalTileSize(int raster_dimension);

/**
 * Checks if the file is an archive and prepends it with the suitable GDAL virtual filesystem prefix.
 *
 * @param file_path The original path to the file.
 * @return File path with a correct prefix if needed.
 */
std::string AdjustFilePath(std::string_view file_path);

std::string ConvertToWkt(std::string_view shp_file_path);

}  // namespace alus
