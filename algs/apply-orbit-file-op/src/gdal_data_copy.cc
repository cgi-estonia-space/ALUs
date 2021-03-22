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
#include "gdal_data_copy.h"

#include "gdal_util.h"

namespace alus {

void GdalDataCopy::CloseDataSets() {
    if (src_dataset_) {
        GDALClose(src_dataset_);
        src_dataset_ = nullptr;
    }
    if (dest_dataset_) {
        GDALClose(dest_dataset_);
        dest_dataset_ = nullptr;
    }
}
GdalDataCopy::GdalDataCopy(std::string_view file_name_src, std::string_view file_name_dst) {
    GDALAllRegister();
    auto const po_driver = GetGDALDriverManager()->GetDriverByName("GTiff");
    CHECK_GDAL_PTR(po_driver);

    src_dataset_ = (GDALDataset*)GDALOpen(file_name_src.data(), GA_ReadOnly);
    CHECK_GDAL_PTR(src_dataset_);

    dest_dataset_ =
        po_driver->CreateCopy(std::string(file_name_dst).c_str(), src_dataset_, FALSE, nullptr, nullptr, nullptr);
    CHECK_GDAL_PTR(dest_dataset_);
}

GdalDataCopy::~GdalDataCopy() { CloseDataSets(); }
}  // namespace alus
