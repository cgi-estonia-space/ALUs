#pragma once

#include <memory>

#include "custom/rectangle.h"
#include "snap-core/datamodel/i_geo_coding.h"
#include "snap-core/subset/abstract_subset_region.h"

namespace alus {
namespace snapengine {
class PixelSubsetRegion : virtual public AbstractSubsetRegion {
private:
    std::shared_ptr<custom::Rectangle> pixel_region_;

protected:
    void ValidateDefaultSize(int default_product_width, int default_product_height,
                             std::string_view exception_message_prefix) override;

public:
    PixelSubsetRegion(int x, int y, int width, int height, int border_pixels);

    PixelSubsetRegion(std::shared_ptr<custom::Rectangle> pixel_region, int border_pixels);

    std::shared_ptr<custom::Rectangle> ComputeProductPixelRegion(
        std::shared_ptr<IGeoCoding> product_default_geo_coding, int default_product_width, int default_product_height,
        bool round_pixel_region) override;

    std::shared_ptr<custom::Rectangle> ComputeBandPixelRegion(
        std::shared_ptr<IGeoCoding> product_default_geo_coding, std::shared_ptr<IGeoCoding> band_default_geo_coding,
        int default_product_width, int default_product_height, int default_band_width, int default_band_height,
        bool round_pixel_region) override;

    std::shared_ptr<custom::Rectangle> GetPixelRegion() { return pixel_region_; }

    static std::shared_ptr<custom::Rectangle> ComputeBandBoundsBasedOnPercent(
        std::shared_ptr<custom::Rectangle> product_bounds, int default_product_width, int default_product_height,
        int default_band_width, int default_band_height);
};
}  // namespace snapengine
}  // namespace alus
