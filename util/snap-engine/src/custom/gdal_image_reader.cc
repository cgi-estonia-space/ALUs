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