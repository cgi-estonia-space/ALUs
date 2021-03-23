/**
 * This file is a filtered duplicate of a SNAP's
 * org.esa.snap.core.datamodel.GeoCoding.java
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
#include <string>

//#include "geo_pos.h"
//#include "pixel_pos.h"

namespace alus {
namespace snapengine {

class GeoPos;
class PixelPos;
class Datum;

using MathTransformWKT = std::string;
using CoordinateReferenceSystemWKT = std::string;
/**
 * The <code>GeoCoding</code> interface provides geo-spatial latitude and longitude information for a given X/Y position
 * of any (two-dimensional) raster.
 * <p> <b> Note: New geo-coding implementations shall implement the abstract class {@link AbstractGeoCoding},
 * instead of implementing this interface.</b>
 *
 * <p>All <code>GeoCoding</code> implementations should override
 * the {@link Object#equals(Object) equals()} and  {@link Object#hashCode() hashCode()} methods.
 *
 * @author Norman Fomferra
 * @version $Revision$ $Date$
 */
class IGeoCoding {
public:
    /**
     * Checks whether or not the longitudes of this geo-coding cross the +/- 180 degree meridian.
     *
     * @return <code>true</code>, if so
     */
    virtual bool IsCrossingMeridianAt180() = 0;

    /**
     * Checks whether or not this geo-coding can determine the pixel position from a geodetic position.
     *
     * @return <code>true</code>, if so
     */
    virtual bool CanGetPixelPos() = 0;

    /**
     * Checks whether or not this geo-coding can determine the geodetic position from a pixel position.
     *
     * @return <code>true</code>, if so
     */
    virtual bool CanGetGeoPos() = 0;

    /**
     * Returns the pixel co-ordinates as x/y for a given geographical position given as lat/lon.
     *
     * @param geoPos   the geographical position as lat/lon in the coordinate system determined by {@link #getGeoCRS()}
     * @param pixelPos an instance of <code>Point</code> to be used as return value. If this parameter is
     *                 <code>null</code>, the method creates a new instance which it then returns.
     * @return the pixel co-ordinates as x/y
     */
    virtual std::shared_ptr<PixelPos> GetPixelPos(const std::shared_ptr<GeoPos>& geo_pos,
                                                  std::shared_ptr<PixelPos>& pixel_pos) = 0;

    /**
     * Returns the latitude and longitude value for a given pixel co-ordinate.
     *
     * @param pixelPos the pixel's co-ordinates given as x,y
     * @param geoPos   an instance of <code>GeoPos</code> to be used as return value. If this parameter is
     *                 <code>null</code>, the method creates a new instance which it then returns.
     * @return the geographical position as lat/lon in the coordinate system determined by {@link #getGeoCRS()}
     */
    virtual std::shared_ptr<GeoPos> GetGeoPos(const std::shared_ptr<PixelPos>& pixel_pos,
                                              std::shared_ptr<GeoPos>& geo_pos) = 0;

    /**
     * Releases all of the resources used by this object instance and all of its owned children. Its primary use is to
     * allow the garbage collector to perform a vanilla job.
     * <p>This method should be called only if it is for sure that this object instance will never be used again. The
     * results of referencing an instance of this class after a call to <code>dispose()</code> are undefined.
     */
    virtual void Dispose() = 0;

    /**
     * @return The image coordinate reference system (CRS). It is usually derived from the base CRS by including
     *         a linear or non-linear transformation from base (geodetic) coordinates to image coordinates.
     */
    //    virtual std::shared_ptr<CoordinateReferenceSystem> GetImageCRS() = 0;
    virtual CoordinateReferenceSystemWKT GetImageCRS() = 0;

    /**
     * @return The map coordinate reference system (CRS). It may be either a geographical CRS (nominal case is
     *         "WGS-84") or a derived projected CRS, e.g. "UTM 32 - North".
     */
    //    virtual std::shared_ptr<CoordinateReferenceSystem> GetMapCRS() = 0;
    virtual CoordinateReferenceSystemWKT GetMapCRS() = 0;

    /**
     * @return The geographical coordinate reference system (CRS). It may be either "WGS-84" (nominal case) or
     *         any other geographical CRS.
     */
    //    virtual std::shared_ptr<CoordinateReferenceSystem> GetGeoCRS() = 0;
    virtual CoordinateReferenceSystemWKT GetGeoCRS() = 0;

    /**
     * @return The math transformation used to convert image coordinates to map coordinates.
     */
    //    virtual std::shared_ptr<MathTransform> GetImageToMapTransform() = 0;
    //    todo: this might need specific gdal object, snap just looks up transformation between 2 systems for this
    //    attribute
    // since java version is able to return WKT version we should be ok to keep it as std::string
    virtual MathTransformWKT GetImageToMapTransform() = 0;

    /**
     * Gets the datum, the reference point or surface against which {@link GeoPos} measurements are made.
     *
     * @return the datum
     * @deprecated use the datum of the associated {@link #getMapCRS() map CRS}.
     */
    [[deprecated]]
    virtual std::shared_ptr<Datum> GetDatum() = 0;

};
}  // namespace snapengine
}  // namespace alus
