#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include <gdal_priv.h>

// todo:move under snapengine/custom?
#include "custom/i_image_reader.h"
#include "custom/rectangle.h"

namespace alus {
namespace snapengine {
namespace custom {

// Placeholder interface for image readers, currently only viable solution is GDAL anyway.
class GdalImageReader : virtual public IImageReader {
private:
    GDALDataset* dataset_{};

public:
    /**
     * avoid tight coupling to data... (swappable sources, targets, types etc..)
     */
    GdalImageReader();

    /**
     * set input path for source to read from
     * @param path_to_band_file
     */
    void SetInputPath(std::string_view path_to_band_file) override;
    /**
     * Make sure std::vector data has correct size before using gdal to fill it
     * if it has wrong size it gets resized
     * @param rectangle
     * @param data
     */
    void ReadSubSampledData(const std::shared_ptr<custom::Rectangle>& rectangle, std::vector<int32_t>& data) override;

    ~GdalImageReader();
};
}  // namespace custom
}  // namespace snapengine
}  // namespace alus