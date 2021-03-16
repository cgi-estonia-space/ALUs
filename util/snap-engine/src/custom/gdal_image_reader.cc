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
#include "custom/gdal_image_reader.h"

#include <cstddef>
#include <stdexcept>

#include "gdal_util.h"

namespace alus {
namespace snapengine {
namespace custom {

GdalImageReader::GdalImageReader() { GDALAllRegister(); }

void GdalImageReader::ReadSubSampledData(const std::shared_ptr<custom::Rectangle>& rectangle,
                                         std::vector<int32_t>& data) {
    // todo:    later add support for subsampled data, this will change parameters for this function
    if (data.size() != static_cast<std::size_t>(rectangle->width * rectangle->height)) {
        data.resize(rectangle->width * rectangle->height);
    }

    CHECK_GDAL_ERROR(dataset_->RasterIO(GF_Read, rectangle->x, rectangle->y, rectangle->width, rectangle->height,
                                        data.data(), rectangle->width, rectangle->height, GDALDataType::GDT_Int32, 1,
                                        nullptr, 0, 0, 0));
}

void GdalImageReader::SetInputPath(std::string_view path_to_band_file) {
    dataset_ = static_cast<GDALDataset*>(GDALOpen(std::string(path_to_band_file).c_str(), GA_ReadOnly));
    CHECK_GDAL_PTR(dataset_);
}

GdalImageReader::~GdalImageReader() {
    if (dataset_) {
        GDALClose(dataset_);
        dataset_ = nullptr;
    }
}

}  // namespace custom
}  // namespace snapengine
}  // namespace alus