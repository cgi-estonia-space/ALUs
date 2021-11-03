#include "gdal_util.h"

#include "general_constants.h"

#include <alus_log.h>

namespace alus {
void GeoTiffWriteFile(GDALDataset* input_dataset, const std::string_view output_file) {
    GDALDriver* output_driver;
    output_driver = GetGdalGeoTiffDriver();

    CHECK_GDAL_PTR(output_driver);

    const std::string output_file_str =
        output_file.find(gdal::constants::GDAL_GTIFF_FILE_EXTENSION) != std::string::npos
            ? output_file.data()
            : std::string(output_file.data()) + gdal::constants::GDAL_GTIFF_FILE_EXTENSION;

    GDALDataset* output_dataset =
        output_driver->CreateCopy(output_file_str.data(), input_dataset, FALSE, nullptr, nullptr, nullptr);
    CHECK_GDAL_PTR(output_dataset);
    GDALClose(output_dataset);
}

std::string findOptimalTileSize(int raster_dimension) {
    // TIFF specification 6.0, Section 15: Tiled Images
    // quote from paragraph TileWidth
    // TileWidth must be a multiple of 16.

    int best = 256;

    // find the best tile size with least empty paddding bytes
    for(int tile_size = 256; tile_size <= 512; tile_size += 16) {
        if(tile_size % raster_dimension == 0) {
            best = tile_size;
            continue;
        }
        const int padding_current = tile_size - raster_dimension % tile_size;
        const int padding_best = best - raster_dimension % best;

        if((raster_dimension % best) != 0 && (padding_current <= padding_best)) {
            best = tile_size;
        }
    }
    return std::to_string(best);
}
}  // namespace alus