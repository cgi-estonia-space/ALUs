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

#include <string_view>
#include <cstdint>
#include <typeinfo>
#include <typeindex>

#include <cpl_error.h>
#include <gdal.h>

namespace alus {
class GdalErrorException final : public std::runtime_error {
   public:
    GdalErrorException(
        CPLErr const errType, CPLErrorNum const errNum, std::string_view errMsg, std::string_view src, int srcLine)
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
    if(std::is_same_v<BufferType, double>){
        return GDALDataType::GDT_Float64;
    }else if(std::is_same_v<BufferType, float>) {
        return GDALDataType::GDT_Float32;
    }else if(std::is_same_v<BufferType, int16_t>){
        return GDALDataType::GDT_Int16;
    }else if(std::is_same_v<BufferType, int32_t>){
        return GDALDataType::GDT_Int32;
    }else if(std::is_same_v<BufferType, Iq16>){
        return GDALDataType::GDT_CInt16;
    }else{
        //todo this function and error can be compile time, but requires refactoring in other places
        throw std::invalid_argument(std::string(typeid(BufferType).name()) + " is not an implemented type for this dataset.");
    }
}
}  // namespace alus

inline void checkGdalError(CPLErr const err, char const* file, int const line) {
    if (err != CE_None) {
        throw alus::GdalErrorException(err, CPLGetLastErrorNo(), CPLGetLastErrorMsg(), file, line);
    }
}

#define CHECK_GDAL_ERROR(err) checkGdalError(err, __FILE__, __LINE__)
#define CHECK_GDAL_PTR(ptr) CHECK_GDAL_ERROR((ptr) == nullptr ? CE_Failure : CE_None)