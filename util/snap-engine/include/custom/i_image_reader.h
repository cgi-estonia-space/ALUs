#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "custom/rectangle.h"

namespace alus {
namespace snapengine {
namespace custom {

// Placeholder interface for image readers, currently only viable solution is GDAL anyway.
class IImageReader {
public:
    /***
     * Just to provide interface between different implementations of band data readers e.g gdal RasterIO
     *
     * @param 2D rectangle area to be read from data
     * @param data container into which data will be placed
     */
    virtual void ReadSubSampledData(const std::shared_ptr<custom::Rectangle>& rectangle,
                                    std::vector<int32_t>& data) = 0;

    virtual void SetInputPath(std::string_view path_to_band_file) = 0;
};
}  // namespace custom
}  // namespace snapengine
}  // namespace alus