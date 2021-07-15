#include "gdal_util.h"

#include "general_constants.h"

#include <alus_log.h>

namespace alus {
void GeoTiffWriteFile(GDALDataset* input_dataset, const std::string_view output_file) {
    GDALDriver* output_driver;
    output_driver = GetGDALDriverManager()->GetDriverByName(utils::constants::GDAL_GTIFF_DRIVER);

    CHECK_GDAL_PTR(output_driver);

    const std::string output_file_str =
        output_file.find(utils::constants::GDAL_GTIFF_FILE_EXTENSION) != std::string::npos
            ? output_file.data()
            : std::string(output_file.data()) + utils::constants::GDAL_GTIFF_FILE_EXTENSION;

    GDALDataset* output_dataset =
        output_driver->CreateCopy(output_file_str.data(), input_dataset, FALSE, nullptr, nullptr, nullptr);
    CHECK_GDAL_PTR(output_dataset);
    GDALClose(output_dataset);
}
}  // namespace alus