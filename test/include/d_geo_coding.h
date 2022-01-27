/**
 * This file is a filtered duplicate of a SNAP's
 * org.esa.snap.core.util.ProductUtilsTest.java
 * ported for native code.
 * Copied from (https://github.com/senbox-org/snap-engine). It was originally stated:
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
#pragma once

#include <memory>

#include "snap-core/core/datamodel/geo_pos.h"
#include "snap-core/core/datamodel/i_geo_coding.h"
#include "snap-core/core/datamodel/pixel_pos.h"

namespace alus::snapengine {

class DGeoCoding : public IGeoCoding {
public:
    DGeoCoding() = default;
    bool IsCrossingMeridianAt180() override { return true; }
    bool CanGetPixelPos() override { return false; }
    bool CanGetGeoPos() override { return true; }
    std::shared_ptr<PixelPos> GetPixelPos([[maybe_unused]] const std::shared_ptr<GeoPos>& geo_pos,
                                          std::shared_ptr<PixelPos>& pixel_pos) override {
        return pixel_pos;
    }
    std::shared_ptr<GeoPos> GetGeoPos(const std::shared_ptr<PixelPos>& pixel_pos,
                                      std::shared_ptr<GeoPos>& geo_pos) override {
        if (geo_pos == nullptr) {
            geo_pos = std::make_shared<GeoPos>();
        }
        geo_pos->lon_ = pixel_pos->x_;
        geo_pos->lat_ = pixel_pos->y_;
        return geo_pos;
    }
    void Dispose() override {}
    CoordinateReferenceSystemWKT GetImageCRS() override { return ""; }
    CoordinateReferenceSystemWKT GetMapCRS() override { return ""; }
    CoordinateReferenceSystemWKT GetGeoCRS() override { return ""; }
    MathTransformWKT GetImageToMapTransform() override { return ""; }
    std::shared_ptr<Datum> GetDatum() override { return nullptr; }
};

}  // namespace alus::snapengine