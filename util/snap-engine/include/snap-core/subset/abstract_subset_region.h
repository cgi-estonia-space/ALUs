#pragma once

#include <memory>
#include <string_view>

#include "custom/rectangle.h"
#include "i_geo_coding.h"

namespace alus {
namespace snapengine {
class AbstractSubsetRegion {
protected:
    int border_pixels_;

    explicit AbstractSubsetRegion(int border_pixels);

    virtual void ValidateDefaultSize(int default_product_width, int default_product_height,
                                     std::string_view exception_message_prefix);

public:
    virtual std::shared_ptr<custom::Rectangle> ComputeProductPixelRegion(
        std::shared_ptr<IGeoCoding> product_default_geo_coding, int default_product_width, int default_product_height,
        bool round_pixel_region) = 0;

    virtual std::shared_ptr<custom::Rectangle> ComputeBandPixelRegion(
        std::shared_ptr<IGeoCoding> product_default_geo_coding, std::shared_ptr<IGeoCoding> band_default_geo_coding,
        int default_product_width, int default_product_height, int default_band_width, int default_band_height,
        bool round_pixel_region) = 0;
};
}  // namespace snapengine
}  // namespace alus
