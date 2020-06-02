#pragma once

#include <string_view>

#include <cpl_error.h>

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
}  // namespace alus

inline void checkGdalError(CPLErr const err, char const* file, int const line) {
    if (err != CE_None) {
        throw alus::GdalErrorException(err, CPLGetLastErrorNo(), CPLGetLastErrorMsg(), file, line);
    }
}

#define CHECK_GDAL_ERROR(err) checkGdalError(err, __FILE__, __LINE__)
#define CHECK_GDAL_PTR(ptr) CHECK_GDAL_ERROR((ptr) == nullptr ? CE_Failure : CE_None)