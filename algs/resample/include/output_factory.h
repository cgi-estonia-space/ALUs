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

#include <future>
#include <memory>
#include <optional>

#include <gdal_priv.h>

#include "alus_log.h"
#include "gdal_util.h"
#include "raster_properties.h"
#include "tyler_the_creator.h"
#include "type_parameter.h"

namespace alus::resample {

constexpr std::string_view DATASET_DOMAIN_METADATA_HINT{"ALUs_DOMAIN"};

std::string CreateResampledTilePath(std::string_view path_stem, size_t tile_x_no, size_t tile_y_no,
                                    std::string_view extension);

struct StorageOutputParameters {
    TypeParameters type_parameters;
    GDALDriver* driver;
    GDALDataType gdal_type;
    OGRSpatialReference srs;
};

inline std::future<void> StoreResampled(
    std::vector<TileProperties> tiles, alus::RasterDimension image_dimension, std::unique_ptr<uint8_t[]> image_data,
    std::string file_path_stem, StorageOutputParameters output_parameters,
    std::vector<std::pair<std::string, std::pair<std::string, std::string>>> metadata,
    std::optional<double> no_data_value) {
    return std::async(std::launch::async, [output_parameters = std::move(output_parameters),
                                           image_data = std::move(image_data), tiles = std::move(tiles),
                                           file_path_stem = std::move(file_path_stem), image_dimension,
                                           metadata = std::move(metadata), no_data_value]() {
        auto* driver = output_parameters.driver;
        const auto* ext = driver->GetMetadataItem(GDAL_DMD_EXTENSION);
        std::string extension{};
        if (ext == nullptr) {
            LOGW << "GDAL driver (" << driver->GetDescription() << ") does not specify file extension";
        } else {
            extension = ext;
        }

        for (auto&& t : tiles) {
            std::string filename = CreateResampledTilePath(file_path_stem, t.tile_no_x, t.tile_no_y, extension);
            auto* ds = driver->Create(filename.c_str(), t.dimension.columnsX, t.dimension.rowsY, 1,
                                      output_parameters.gdal_type, nullptr);
            CHECK_GDAL_PTR(ds);

            auto ds_close = [](GDALDataset* ds) { GDALClose(ds); };
            std::unique_ptr<GDALDataset, decltype(ds_close)> ds_guard(ds, ds_close);

            GDALRasterBand* band = ds_guard->GetRasterBand(gdal::constants::GDAL_DEFAULT_RASTER_BAND);
            for (int row{}; row < t.dimension.rowsY; row++) {
                const size_t offset =
                    ((t.offset.y + row) * image_dimension.columnsX) * output_parameters.type_parameters.size_bytes +
                    t.offset.x * output_parameters.type_parameters.size_bytes;
                CHECK_GDAL_ERROR(band->RasterIO(GF_Write, 0, row, t.dimension.columnsX, 1, image_data.get() + offset,
                                                t.dimension.columnsX, 1, output_parameters.gdal_type, 0, 0));
            }

            for (const auto& [domain, md_item] : metadata) {
                if (domain != DATASET_DOMAIN_METADATA_HINT) {
                    CHECK_GDAL_ERROR(band->SetMetadataItem(md_item.first.data(), md_item.second.data(), domain.data()));
                } else {
                    CHECK_GDAL_ERROR(ds_guard->SetMetadataItem(md_item.first.data(), md_item.second.data()));
                }
            }

            if (no_data_value.has_value()) {
                CHECK_GDAL_ERROR(band->SetNoDataValue(no_data_value.value()));
            }

            CHECK_GDAL_ERROR(ds_guard->SetSpatialRef(&output_parameters.srs));
            auto gt = GeoTransformConstruct::ConvertToGdal(t.gt);
            CHECK_GDAL_ERROR(ds_guard->SetGeoTransform(gt.data()));
        }
    });
}

}  // namespace alus::resample