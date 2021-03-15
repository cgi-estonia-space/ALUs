/**
 * This file is a filtered duplicate of a SNAP's org.esa.snap.core.util.ProductUtils.java
 * ported for native code.
 * Copied from a snap-engine's(https://github.com/senbox-org/snap-engine) repository originally stated
 * to be implemented by "Copyright (C) 2010 Brockmann Consult GmbH (info@brockmann-consult.de)"
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

//#include <geos/geom/Geometry.h>
//#include <geos/geom/GeometryFactory.h>
//#include <geos/geom/Polygon.h>

#include "custom/rectangle.h"
#include "i_geo_coding.h"

namespace alus {
namespace snapengine {

struct DistanceHeading {
    double distance;
    double heading1;
    double heading2;
};

struct WGS84 {
    static constexpr double a = 6378137.0;                          // m
    static constexpr double b = 6356752.3142451794975639665996337;  // 6356752.31424518; // m
    static constexpr double earth_flat_coef = 1.0 / ((a - b) / a);  // 298.257223563;
    static constexpr double e2 = 2.0 / earth_flat_coef - 1.0 / (earth_flat_coef * earth_flat_coef);
    static constexpr double e2inv = 1 - WGS84::e2;
    static constexpr double ep2 = e2 / (1 - e2);
};

class GeoUtils {
private:
    static constexpr double EPS5 = 1e-5;
    static constexpr double EPS = 1e-10;
    // GeneralPath[]
    //    static std::vector<std::string> CreateGeoBoundaryPaths(std::shared_ptr<IGeoCoding> geo_coding,
    //                                                           std::shared_ptr<custom::Rectangle> region, int
    //                                                           step, bool use_pixel_center);
    //    static std::vector<GeoPos> CreateGeoBoundary(std::shared_ptr<IGeoCoding> geo_coding,
    //                                                 std::shared_ptr<custom::Rectangle> region, int step,
    //                                                 bool use_pixel_center);
    //
    //    static std::shared_ptr<geos::geom::Polygon> ConvertAwtPathToJtsPolygon(
    //        std::vector<double> path, std::unique_ptr<geos::geom::GeometryFactory> factory);
    //
public:
    //    static std::unique_ptr<geos::geom::Geometry> ComputeGeometryUsingPixelRegion(
    //        std::shared_ptr<IGeoCoding> raster_geo_coding, std::shared_ptr<custom::Rectangle> pixel_region);
    //
    //    static std::shared_ptr<custom::Rectangle> ComputePixelRegionUsingGeometry(
    //        const std::shared_ptr<IGeoCoding>& raster_geo_coding, int raster_width, int raster_height,
    //        const std::shared_ptr<geos::geom::Geometry>& geometry_region, int num_border_pixels, bool
    //        round_pixel_region);

    /**
     * Given starting and end points
     * calculate distance in meters and initial headings from start to
     * end (return variable head1),
     * and from end to start point (return variable head2)
     *
     * @param pos1 first position
     * @param pos2 second position
     * @return distance and heading
     * dist:	distance in m
     * head1:	azimuth in degrees mesured in the direction North east south west
     * 			from (lon1,lat1) to (lon2, lat2)
     * head2:	azimuth in degrees mesured in the direction North east south west
     * 			from (lon2,lat2) to (lon1, lat1)
     */
    static std::shared_ptr<DistanceHeading> VincentyInverse(std::shared_ptr<GeoPos> pos1, std::shared_ptr<GeoPos> pos2);
};
}  // namespace snapengine
}  // namespace alus
