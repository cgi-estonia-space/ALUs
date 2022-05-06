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
#include "custom/gdal_image_writer.h"

#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "gdal_util.h"
#include "general_constants.h"

namespace alus::snapengine::custom {

void GdalImageWriter::WriteSubSampledData(const custom::Rectangle& rectangle, std::vector<float>& data,
                                          int band_index) {
    if (data.size() > static_cast<std::size_t>(rectangle.width) * static_cast<size_t>(rectangle.height)) {
        throw std::runtime_error("Buffer overflow");
    }
    CHECK_GDAL_ERROR(dataset_->GetRasterBand(band_index)
                         ->RasterIO(GF_Write, rectangle.x, rectangle.y, rectangle.width, rectangle.height, data.data(),
                                    rectangle.width, rectangle.height, GDALDataType::GDT_Float32, 0, 0));
}
void GdalImageWriter::WriteSubSampledData(const alus::Rectangle& rectangle, std::vector<float>& data, int band_index) {
    custom::Rectangle region(rectangle);
    WriteSubSampledData(region, data, band_index);
}

void GdalImageWriter::Open(std::string_view path_to_band_file, int raster_size_x, int raster_size_y,
                           std::vector<double> affine_geo_transform_out, const std::string_view data_projection_out,
                           bool in_memory_file) {
    auto* const po_driver = in_memory_file ? GetGdalMemDriver() : GetGdalGeoTiffDriver();

    CHECK_GDAL_PTR(po_driver);
    // po_driver reference gets checked by guard
    dataset_ = po_driver->Create(std::string(path_to_band_file).c_str(), raster_size_x, raster_size_y, 1, GDT_Float32,
                                 nullptr);
    CHECK_GDAL_PTR(dataset_);
    if (!affine_geo_transform_out.empty()) {
        CHECK_GDAL_ERROR(dataset_->SetGeoTransform(affine_geo_transform_out.data()));
    }
    CHECK_GDAL_ERROR(dataset_->SetProjection(data_projection_out.data()));
}

void GdalImageWriter::Close() {
    GDALClose(dataset_);
    dataset_ = nullptr;
}

GdalImageWriter::~GdalImageWriter() {
    GDALClose(dataset_);
    dataset_ = nullptr;
}

}  // namespace alus::snapengine::custom