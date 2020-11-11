#pragma once

#include <string_view>

#include <gdal_priv.h>

namespace alus {
class GdalDataCopy {
private:
    GDALDataset* src_dataset_{};
    GDALDataset* dest_dataset_{};
    void CloseDataSets();

public:
    GdalDataCopy(std::string_view file_name_src, std::string_view file_name_dst);
    ~GdalDataCopy();
};
}  // namespace alus
