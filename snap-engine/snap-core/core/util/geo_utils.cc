/**
 * This file is a filtered duplicate of a SNAP's
 * org.esa.snap.core.util.GeoUtils.java
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
#include "snap-core/core/util/geo_utils.h"

#include <cmath>
#include <memory>
//#include <vector>
//
//#include <geos/geom/Coordinate.h>
//#include <geos/geom/GeometryFactory.h>
//#include <geos/geom/Polygon.h>

#include "product_utils.h"
#include "snap-engine-utilities/engine-utilities/eo/constants.h"

namespace alus::snapengine {

// todo: add when needed
// std::unique_ptr<geos::geom::Geometry> GeoUtils::ComputeGeometryUsingPixelRegion(
//    std::shared_ptr<GeoCoding> raster_geo_coding, std::shared_ptr<alus::cpp17::Rectangle> pixel_region) {
//    if (pixel_region == nullptr) {
//        throw std::runtime_error("The pixel region is null.");
//    }
//    int step = std::min(pixel_region->width, pixel_region->height) / 8;
//    //        todo:need to replace awt logic here (hardcoded placeholder to continue)
//    std::vector<int> paths(3);
//    //        GeneralPath[] paths = createGeoBoundaryPaths(rasterGeoCoding, pixelRegion, step, false);
//    std::vector<std::unique_ptr<geos::geom::Polygon>> polygons(paths.size());
//    auto factory = std::make_unique<geos::geom::GeometryFactory>();
//    for (size_t i = 0; i < paths.size(); i++) {
//        polygons.at(i) = ConvertAwtPathToJtsPolygon(paths.at(i), factory);
//    }
//    if (polygons.size() == 1) {
//        return std::dynamic_pointer_cast<geos::geom::Geometry>(polygons.at(0));
//    } else {
//        return factory->createMultiPolygon(polygons);
//    }
//}

// std::vector<std::string> GeoUtils::CreateGeoBoundaryPaths(std::shared_ptr<GeoCoding> geo_coding,
//                                                          std::shared_ptr<alus::cpp17::Rectangle> region, int step,
//                                                          bool use_pixel_center) {
//    if (geo_coding == nullptr) {
//        throw std::runtime_error("The geo coding is null.");
//    }
//    if (region == nullptr) {
//        throw std::runtime_error("The region is null.");
//    }
//    std::vector<GeoPos> geo_points = CreateGeoBoundary(geo_coding, region, step, use_pixel_center);
//    ProductUtils::NormalizeGeoPolygon(geo_points);
//    final ArrayList<GeneralPath> path_list = AssemblePathList(geo_points);
//    return pathList.toArray(new GeneralPath[pathList.size()]);
//}
// std::vector<GeoPos> GeoUtils::CreateGeoBoundary(std::shared_ptr<GeoCoding> geo_coding,
//                                                std::shared_ptr<alus::cpp17::Rectangle> region, int step, bool
//                                                use_pixel_center) {
//    if (geo_coding == nullptr) {
//        throw std::runtime_error("The geo coding is null.");
//    }
//    if (region == nullptr) {
//        throw std::runtime_error("The region is null.");
//    }
//    std::vector<PixelPos> points = GeoUtils::CreatePixelBoundaryFromRect(region, step, use_pixel_center);
//    std::vector<GeoPos> geo_points(points.size());
//    for (PixelPos pixel_pos : points) {
//        auto gc_geo_pos = geo_coding->GetGeoPos(pixel_pos, nullptr);
//        geo_points.emplace_back(gc_geo_pos);
//    }
//    return geo_points;
//}
// std::shared_ptr<alus::cpp17::Rectangle> GeoUtils::ComputePixelRegionUsingGeometry(
//    const std::shared_ptr<IGeoCoding>& raster_geo_coding, int raster_width, int raster_height,
//    const std::shared_ptr<geos::geom::Geometry>& geometry_region, int num_border_pixels, bool round_pixel_region) {
//    std::shared_ptr<geos::geom::Geometry> raster_geometry =
//        ComputeRasterGeometry(raster_geo_coding, raster_width, raster_height);
//    std::shared_ptr<geos::geom::Geometry> region_intersection = geometry_region->intersection(raster_geometry);
//    if (region_intersection->isEmpty()) {
//        return std::make_shared<alus::cpp17::Rectangle>();  // the intersection is empty
//    }
//    GeoUtils::PixelRegionFinder pixel_region_finder =
//        std::make_unique<GeoUtils.PixelRegionFinder>(rasterGeoCoding, roundPixelRegion);
//    region_intersection.apply(pixel_region_finder);
//    std::shared_ptr<alus::cpp17::Rectangle> pixel_region = pixel_region_finder.getPixelRegion();
//    pixel_region.grow(num_border_pixels, num_border_pixels);
//    return pixel_region.intersection(std::make_shared<alus::cpp17::Rectangle>(raster_width, raster_height));
//}
// std::shared_ptr<geos::geom::Polygon> GeoUtils::ConvertAwtPathToJtsPolygon(
//    std::vector<double> path, std::unique_ptr<geos::geom::GeometryFactory> factory) {
//    PathIterator pathIterator = path.getPathIterator(null);
//    ArrayList<double[]> coordList = new ArrayList<>();
//    int lastOpenIndex = 0;
//    while (!path_iterator.isDone()) {
//        std::vector<double> coords(6);
//        int seg_type = path_iterator.current_segment(coords);
//        if (seg_type == PathIterator.SEG_CLOSE) {
//            // we should only detect a single SEG_CLOSE
//            coord_list.add(coord_list.at(last_open_index));
//            lastOpenIndex = coord_list.size();
//        } else {
//            coord_list.add(coords);
//        }
//        pathIterator.next();
//    }
//    std::vector<geos::geom::Coordinate> coordinates(coord_list.size());
//    for (size_t i1 = 0; i1 < coordinates.size(); i1++) {
//        std::vector<double> coord = coord_list.at(i1);
//        coordinates.at(i1) = new Coordinate(coord.at(0), coord.at(1));
//    }
//
//    return factory->createPolygon(factory->createLinearRing(coordinates), nullptr);
//}

std::shared_ptr<DistanceHeading> GeoUtils::VincentyInverse(const std::shared_ptr<GeoPos>& pos1,
                                                           const std::shared_ptr<GeoPos>& pos2) {
    std::shared_ptr<DistanceHeading> output = std::make_shared<DistanceHeading>();
    double lat1 = pos1->lat_;
    double lon1 = pos1->lon_;
    double lat2 = pos2->lat_;
    double lon2 = pos2->lon_;

    if ((std::abs(lon1 - lon2) < EPS5) && (std::abs(lat1 - lat2) < EPS5)) {
        output->distance = 0;
        output->heading1 = -1;
        output->heading2 = -1;
        return output;
    }

    lat1 *= eo::constants::DTOR;
    lat2 *= eo::constants::DTOR;
    lon1 *= eo::constants::DTOR;
    lon2 *= eo::constants::DTOR;

    // Model WGS84:
    //    f=1/298.25722210;	// flattening
    double f = 0.0;  // defF;

    double r = 1 - f;
    double tu_1 = r * std::tan(lat1);
    double tu_2 = r * std::tan(lat2);
    double cu_1 = 1.0 / std::sqrt(tu_1 * tu_1 + 1.0);
    double su_1 = cu_1 * tu_1;
    double cu_2 = 1.0 / std::sqrt(tu_2 * tu_2 + 1.0);
    double s = cu_1 * cu_2;
    double baz = s * tu_2;
    double faz = baz * tu_1;
    double x = lon2 - lon1;

    double sx, cx, sy, cy, y, sa, c_2_a, cz, e, c,
        d;  // NOLINT (too many variables are being declared with default initialisation)
    do {
        sx = std::sin(x);
        cx = std::cos(x);
        tu_1 = cu_2 * sx;
        tu_2 = baz - su_1 * cu_2 * cx;
        sy = std::sqrt(tu_1 * tu_1 + tu_2 * tu_2);
        cy = s * cx + faz;
        y = std::atan2(sy, cy);
        sa = s * sx / sy;
        c_2_a = -sa * sa + 1.;
        cz = faz + faz;
        if (c_2_a > 0.) {
            cz = -cz / c_2_a + cy;
        }
        e = cz * cz * 2. - 1.;
        c = ((-3. * c_2_a + 4.) * f + 4.) * c_2_a * f / 16.;  // NOLINT
        d = x;
        x = ((e * cy * c + cz) * sy * c + y) * sa;
        x = (1. - c) * x * f + lon2 - lon1;
    } while (std::abs(d - x) > (0.01));  // NOLINT

    faz = std::atan2(tu_1, tu_2);
    baz = std::atan2(cu_1 * sx, baz * cx - su_1 * cu_2) + eo::constants::PI;
    x = std::sqrt((1. / r / r - 1.) * c_2_a + 1.) + 1.;
    x = (x - 2.) / x;
    c = 1. - x;
    c = (x * x / 4. + 1.) / c;     // NOLINT
    d = (0.375 * x * x - 1.) * x;  // NOLINT
    x = e * cy;
    s = 1. - e - e;
    s = ((((sy * sy * 4. - 3.) * s * cz * d / 6. - x) * d / 4. + cz) * sy * d + y) * c * WGS84::A * r;  // NOLINT

    output->distance = s;
    output->heading1 = faz * eo::constants::RTOD;
    output->heading2 = baz * eo::constants::RTOD;

    while (output->heading1 < 0) output->heading1 += 360;  // NOLINT
    while (output->heading2 < 0) output->heading2 += 360;  // NOLINT

    return output;
}

}  // namespace alus::snapengine