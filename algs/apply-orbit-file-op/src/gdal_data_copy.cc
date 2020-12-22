#include "gdal_data_copy.h"

#include "gdal_util.hpp"

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
