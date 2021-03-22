/**
 * This file is a filtered duplicate of a SNAP's
 * org.esa.snap.core.datamodel.AbstractGeoCoding.java
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

#include "geo_pos.h"
#include "pixel_pos.h"

#include "snap-core/datamodel/i_geo_coding.h"

namespace alus {
namespace snapengine {

// pre-declare
class IScene;
class ProductSubsetDef;

/**
 * <code>AbstractGeoCoding</code> is the base class of all geo-coding implementation.
 * <p> <b> Note:</b> New geo-coding implementations shall implement this abstract class, instead of
 * implementing the interface {@link GeoCoding}.
 *
 * original java version author: Marco Peters
 */
class AbstractGeoCoding : public virtual IGeoCoding {
private:
    CoordinateReferenceSystemWKT image_c_r_s_;
    CoordinateReferenceSystemWKT map_c_r_s_;
    CoordinateReferenceSystemWKT geo_c_r_s_;
    MathTransformWKT image_2_map_;

protected:
    //     todo: use correct WGS84 string when used (instead of DefaultGeographicCRS.WGS84)
    /**
     * Default constructor. Sets WGS84 as both the geographic CRS and map CRS.
     */
    AbstractGeoCoding() : AbstractGeoCoding("DefaultGeographicCRS.WGS84"){};

    /**
     * Constructor.
     *
     * @param geoCRS The CRS to be used as both the geographic CRS and map CRS.
     */
    AbstractGeoCoding(const CoordinateReferenceSystemWKT& geo_c_r_s) {
        SetGeoCRS(geo_c_r_s);
        SetMapCRS(geo_c_r_s);
        SetImageCRS(GetMapCRS());
    }

    void SetImageCRS(const CoordinateReferenceSystemWKT& image_c_r_s) {
        //        todo: current wkt string can't be null anyway (can be empty)
        //        Assert::NotNull(image_c_r_s, "imageCRS");
        image_c_r_s_ = image_c_r_s;
    }

    void SetMapCRS(const CoordinateReferenceSystemWKT& map_c_r_s) { map_c_r_s_ = map_c_r_s; }

    void SetGeoCRS(const CoordinateReferenceSystemWKT& geo_c_r_s) { geo_c_r_s_ = geo_c_r_s; }

    //    static DefaultDerivedCRS createImageCRS(CoordinateReferenceSystem baseCRS,
    //                                                  MathTransform baseToDerivedTransform) {
    //        return new DefaultDerivedCRS("Image CS based on " + baseCRS.getName(),
    //                                     baseCRS,
    //                                     baseToDerivedTransform,
    //                                     DefaultCartesianCS.DISPLAY);
    //    }

public:
    /**
     * Transfers the geo-coding of the {@link Scene srcScene} to the {@link Scene destScene} with respect to the given
     * {@link ProductSubsetDef subsetDef}.
     *
     * @param srcScene  the source scene
     * @param destScene the destination scene
     * @param subsetDef the definition of the subset, may be <code>null</code>
     * @return true, if the geo-coding could be transferred.
     */
    virtual bool TransferGeoCoding(const std::shared_ptr<IScene>& src_scene, const std::shared_ptr<IScene>& dest_scene,
                                   const std::shared_ptr<ProductSubsetDef>& subset_def) = 0;

    CoordinateReferenceSystemWKT GetImageCRS() override { return image_c_r_s_; }
    CoordinateReferenceSystemWKT GetMapCRS() override { return map_c_r_s_; }
    CoordinateReferenceSystemWKT GetGeoCRS() override { return geo_c_r_s_; }

    MathTransformWKT GetImageToMapTransform() override;

};
}  // namespace snapengine
}  // namespace alus
