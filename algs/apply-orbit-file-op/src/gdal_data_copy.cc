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

#include <gdal_priv.h>

#include "gdal_util.h"

namespace alus {

void GdalDataCopy(const char* file_name_src, const char* file_name_dst) {
    auto const po_driver = GetGDALDriverManager()->GetDriverByName("GTiff");
    CHECK_GDAL_PTR(po_driver);

    GDALDataset* src_dataset = (GDALDataset*)GDALOpen(file_name_src, GA_ReadOnly);;
    GDALDataset* dest_dataset = nullptr;

    if(src_dataset != nullptr) {
        dest_dataset =
            po_driver->CreateCopy(file_name_dst, src_dataset, FALSE, nullptr, nullptr, nullptr);
    }

    GDALClose(src_dataset); // closing nullptr is fine
    GDALClose(dest_dataset);

    CHECK_GDAL_PTR(src_dataset); // throw the exception if either op failed
    CHECK_GDAL_PTR(dest_dataset);
}
}  // namespace alus
