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

namespace alus {
namespace gdal::constants {
// GDAL driver constants
constexpr char GDAL_MEM_DRIVER[]{"MEM"};
constexpr char GDAL_GTIFF_DRIVER[]{"GTiff"};
constexpr char GDAL_GTIFF_FILE_EXTENSION[]{".tif"};
constexpr int GDAL_DEFAULT_RASTER_BAND{1};

constexpr std::string_view GDAL_ZIP_PREFIX{"/vsizip/"};
constexpr std::string_view GDAL_GZIP_PREFIX{"/vsigzip/"};
constexpr std::string_view GDAL_TAR_PREFIX{"/vsitar/"};

constexpr std::string_view ZIP_EXTENSION{".zip"};
constexpr std::string_view GZIP_EXTENSION{".gz"};
constexpr std::string_view TAR_EXTENSION{".tar"};
constexpr std::string_view TGZ_EXTENSION{
    ".tgz"};  // TODO: .tar.gz is not supported yet (would require additional checks in AdjustFilePath()
}  // namespace constants

inline GDALDriver* GetGdalMemDriver() {
    return GetGDALDriverManager()->GetDriverByName(gdal::constants::GDAL_MEM_DRIVER);
}

inline GDALDriver* GetGdalGeoTiffDriver() {
    return GetGDALDriverManager()->GetDriverByName(gdal::constants::GDAL_GTIFF_DRIVER);
}

class GdalErrorException final : public std::runtime_error {
public:
    GdalErrorException(CPLErr const errType, CPLErrorNum const errNum, std::string_view errMsg, std::string_view src,
                       int srcLine)
        : std::runtime_error("GDAL error no " + std::to_string(static_cast<int>(errNum)) + " type " +
                             std::to_string(static_cast<int>(errType)) + " - '" + std::string{errMsg} + "' at " +
                             std::string{src} + ":" + std::to_string(srcLine)),
          gdalError{errNum},
          file{src},
          line{srcLine} {}

    [[nodiscard]] CPLErrorNum getGdalError() const { return gdalError; }
    [[nodiscard]] std::string_view getSource() const { return file; }
    [[nodiscard]] int getLine() const { return line; }

private:
    CPLErrorNum const gdalError;
    std::string file;
    int const line;
};

struct Iq16 {
    int16_t i;
    int16_t q;
};
static_assert(sizeof(Iq16) == 4, "Do no alter the memory layout of this structure!");

template <typename BufferType>
GDALDataType FindGdalDataType() {
    if (std::is_same_v<BufferType, double>) {
        return GDALDataType::GDT_Float64;
    } else if (std::is_same_v<BufferType, float>) {
        return GDALDataType::GDT_Float32;
    } else if (std::is_same_v<BufferType, int16_t>) {
        return GDALDataType::GDT_Int16;
    } else if (std::is_same_v<BufferType, int32_t>) {
        return GDALDataType::GDT_Int32;
    } else if (std::is_same_v<BufferType, Iq16>) {
        return GDALDataType::GDT_CInt16;
    } else {
        // todo this function and error can be compile time, but requires refactoring in other places
        throw std::invalid_argument(std::string(typeid(BufferType).name()) +
                                    " is not an implemented type for this dataset.");
    }
}

void GeoTiffWriteFile(GDALDataset* input_dataset, const std::string_view output_file);

std::string findOptimalTileSize(int raster_dimension);

/**
 * Checks if the file is an archive and prepends it with the suitable GDAL virtual filesystem prefix.
 *
 * @param file_path The original path to the file.
 * @return File path with a correct prefix if needed.
 */
std::string AdjustFilePath(std::string_view file_path);
}  // namespace alus::gdal

inline void checkGdalError(CPLErr const err, char const* file, int const line) {
    if (err != CE_None) {
        throw alus::GdalErrorException(err, CPLGetLastErrorNo(), CPLGetLastErrorMsg(), file, line);
    }
}

#define CHECK_GDAL_ERROR(err) checkGdalError(err, __FILE__, __LINE__)
#define CHECK_GDAL_PTR(ptr) CHECK_GDAL_ERROR((ptr) == nullptr ? CE_Failure : CE_None)
