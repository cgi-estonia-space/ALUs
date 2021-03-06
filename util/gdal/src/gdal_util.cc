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
#include "gdal_util.h"

#include "general_constants.h"

#include <alus_log.h>

#include <boost/filesystem/path.hpp>

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

std::string FindOptimalTileSize(int raster_dimension) {
    // TIFF specification 6.0, Section 15: Tiled Images
    // quote from paragraph TileWidth
    // TileWidth must be a multiple of 16.

    const int default_tile_size{256};
    int best = default_tile_size;

    // find the best tile size with least empty paddding bytes
    for (int tile_size = 256; tile_size <= 512; tile_size += 16) {  // NOLINT
        if (tile_size % raster_dimension == 0) {
            best = tile_size;
            continue;
        }
        const int padding_current = tile_size - raster_dimension % tile_size;
        const int padding_best = best - raster_dimension % best;

        if ((raster_dimension % best) != 0 && (padding_current <= padding_best)) {
            best = tile_size;
        }
    }
    return std::to_string(best);
}

std::string AdjustFilePath(std::string_view file_path) {
    const auto file_extension = boost::filesystem::path(file_path.data()).extension().string();
    if (file_extension == gdal::constants::ZIP_EXTENSION) {
        return std::string(gdal::constants::GDAL_ZIP_PREFIX).append(file_path);
    }

    if (file_extension == gdal::constants::GZIP_EXTENSION) {
        return std::string(gdal::constants::GDAL_GZIP_PREFIX).append(file_path);
    }

    if (file_extension == gdal::constants::TAR_EXTENSION || file_extension == gdal::constants::TGZ_EXTENSION) {
        return std::string(gdal::constants::GDAL_TAR_PREFIX).append(file_path);
    }

    throw std::invalid_argument(std::string("Unknown file extension: ").append(file_extension));
}
}  // namespace alus