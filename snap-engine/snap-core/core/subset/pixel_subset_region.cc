/**
 * This file is a filtered duplicate of a SNAP's
 * org.esa.snap.core.subset.PixelSubsetRegion.java
 * ported for native code.
 *
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
#include "snap-core/core/subset/pixel_subset_region.h"

#include <stdexcept>

// keeping this for future reference (in case we port vector)
//#include <geos/geom/Geometry.h>

#include "snap-core/core/util/geo_utils.h"

namespace alus::snapengine {

std::shared_ptr<custom::Rectangle> PixelSubsetRegion::ComputeBandBoundsBasedOnPercent(
    const std::shared_ptr<custom::Rectangle>& product_bounds, int default_product_width, int default_product_height,
    int default_band_width, int default_band_height) {
    float product_offset_x_percent = static_cast<float>(product_bounds->x) / static_cast<float>(default_product_width);
    float product_offset_y_percent = static_cast<float>(product_bounds->y) / static_cast<float>(default_product_height);
    float product_width_percent = static_cast<float>(product_bounds->width) / static_cast<float>(default_product_width);
    float product_height_percent =
        static_cast<float>(product_bounds->height) / static_cast<float>(default_product_height);
    int band_offset_x = static_cast<int>(product_offset_x_percent * static_cast<float>(default_band_width));
    int band_offset_y = static_cast<int>(product_offset_y_percent * static_cast<float>(default_band_height));
    int band_width = static_cast<int>(product_width_percent * static_cast<float>(default_band_width));
    int band_height = static_cast<int>(product_height_percent * static_cast<float>(default_band_height));
    return std::make_shared<custom::Rectangle>(band_offset_x, band_offset_y, band_width, band_height);
}

std::shared_ptr<custom::Rectangle> PixelSubsetRegion::ComputeBandPixelRegion(
    [[maybe_unused]] std::shared_ptr<IGeoCoding> product_default_geo_coding,
    [[maybe_unused]] std::shared_ptr<IGeoCoding> band_default_geo_coding, [[maybe_unused]] int default_product_width,
    [[maybe_unused]] int default_product_height, [[maybe_unused]] int default_band_width,
    [[maybe_unused]] int default_band_height, [[maybe_unused]] bool round_pixel_region) {
    throw std::runtime_error("not yet ported/supported");
    //    ValidateDefaultSize(default_product_width, default_product_height, "The default product");
    //    // test if the band width and band height > 0
    //    AbstractSubsetRegion::ValidateDefaultSize(default_band_width, default_band_height, "The default band");
    //
    //    if (default_product_width != default_band_width || default_product_height != default_band_height) {
    //        // the product is multisize
    //        if (product_default_geo_coding != nullptr && band_default_geo_coding != nullptr) {
    //            std::shared_ptr<geos::geom::Geometry> product_geometry_region =
    //                GeoUtils::ComputeGeometryUsingPixelRegion(product_default_geo_coding, pixel_region_);
    //            return GeoUtils::ComputePixelRegionUsingGeometry(band_default_geo_coding, default_band_width,
    //                                                             default_band_height, product_geometry_region,
    //                                                             border_pixels_, round_pixel_region);
    //        }
    //        return ComputeBandBoundsBasedOnPercent(pixel_region_, default_product_width, default_product_height,
    //                                               default_band_width, default_band_height);
    //    }
    //    return pixel_region_;
}

std::shared_ptr<custom::Rectangle> PixelSubsetRegion::ComputeProductPixelRegion(
    [[maybe_unused]] std::shared_ptr<IGeoCoding> product_default_geo_coding, int default_product_width,
    int default_product_height, [[maybe_unused]] bool round_pixel_region) {
    ValidateDefaultSize(default_product_width, default_product_height, "The default product");
    return pixel_region_;
}

PixelSubsetRegion::PixelSubsetRegion(const std::shared_ptr<custom::Rectangle>& pixel_region, int border_pixels)
    : AbstractSubsetRegion(border_pixels) {
    if (pixel_region == nullptr) {
        throw std::runtime_error("The pixel region is null.");
    }
    if (pixel_region->x < 0 || pixel_region->y < 0 || pixel_region->width < 1 || pixel_region->height < 1) {
        //        throw std::invalid_argument("The pixel region '" + pixel_region + "' is invalid.");
        //        todo: override to provide region
        throw std::invalid_argument("The pixel region 'PIXEL REGION TO STR' is invalid.");
    }
    pixel_region_ = pixel_region;
}

void PixelSubsetRegion::ValidateDefaultSize(int default_product_width, int default_product_height,
                                            std::string_view exception_message_prefix) {
    AbstractSubsetRegion::ValidateDefaultSize(default_product_width, default_product_height, exception_message_prefix);

    if (default_product_width < pixel_region_->width) {
        throw std::invalid_argument(
            std::string(exception_message_prefix) + " width '" + std::to_string(default_product_width) +
            "' must be greater or equal than the pixel region width " + std::to_string(pixel_region_->width) + ".");
    }
    if (default_product_height < pixel_region_->height) {
        throw std::invalid_argument(
            std::string(exception_message_prefix) + " height '" + std::to_string(default_product_height) +
            "' must be greater or equal than the pixel region height " + std::to_string(pixel_region_->height) + ".");
    }
}

PixelSubsetRegion::PixelSubsetRegion(int x, int y, int width, int height, int border_pixels)
    : AbstractSubsetRegion(border_pixels) {
    if (x < 0 || y < 0 || width < 1 || height < 1) {
        throw std::invalid_argument("The pixel region 'x=" + std::to_string(x) + ", y=" + std::to_string(y) +
                                    ", width=" + std::to_string(width) + ",height=" + std::to_string(height) +
                                    "' is invalid.");
    }
    pixel_region_ = std::make_shared<custom::Rectangle>(x, y, width, height);
}

}  // namespace alus::snapengine