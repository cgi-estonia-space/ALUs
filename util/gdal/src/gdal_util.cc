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

#include <filesystem>
#include <stdexcept>

#include <ogrsf_frmts.h>


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
    const auto file_extension = std::filesystem::path(file_path.data()).extension().string();
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


std::string ConvertToWkt(std::string_view shp_file_path) {

    auto ds = (GDALDataset*)GDALOpenEx( shp_file_path.data(), GDAL_OF_VECTOR, NULL, NULL, NULL );
    CHECK_GDAL_PTR(ds);

    auto layers = ds->GetLayers();
    if (layers.size() != 1) {
        throw std::invalid_argument("Expecting a shapefile with 1 layer exact.");
    }

    auto aoi_layer = layers[0];
    if (aoi_layer->GetFeatureCount() == 0) {
        throw std::invalid_argument("Expecting a shapefile with some features.");
    }
    auto* feat = aoi_layer->GetNextFeature();
    CHECK_GDAL_PTR(feat);
    auto gref = feat->GetGeometryRef();
    CHECK_GDAL_PTR(gref);
    char* wkt_shp;
    CHECK_OGR_ERROR(gref->exportToWkt(&wkt_shp));
    auto cpl_free = [](char* csl) { CPLFree(csl); };
    std::unique_ptr<char, decltype(cpl_free)> guard(wkt_shp, cpl_free);

    return std::string(wkt_shp);
}
}  // namespace alus