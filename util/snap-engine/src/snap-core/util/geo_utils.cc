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
#include "snap-core/util/geo_utils.h"

#include <cmath>
#include <memory>
//#include <vector>
//
//#include <geos/geom/Coordinate.h>
//#include <geos/geom/GeometryFactory.h>
//#include <geos/geom/Polygon.h>

#include "product_utils.h"
#include "snap-engine-utilities/eo/constants.h"

namespace alus {
namespace snapengine {

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

std::shared_ptr<DistanceHeading> GeoUtils::VincentyInverse(std::shared_ptr<GeoPos> pos1, std::shared_ptr<GeoPos> pos2) {
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

    lat1 *= Constants::DTOR;
    lat2 *= Constants::DTOR;
    lon1 *= Constants::DTOR;
    lon2 *= Constants::DTOR;

    // Model WGS84:
    //    F=1/298.25722210;	// flattening
    double F = 0.0;  // defF;

    double R = 1 - F;
    double TU1 = R * std::tan(lat1);
    double TU2 = R * std::tan(lat2);
    double CU1 = 1.0 / std::sqrt(TU1 * TU1 + 1.0);
    double SU1 = CU1 * TU1;
    double CU2 = 1.0 / std::sqrt(TU2 * TU2 + 1.0);
    double S = CU1 * CU2;
    double BAZ = S * TU2;
    double FAZ = BAZ * TU1;
    double X = lon2 - lon1;

    double SX, CX, SY, CY, Y, SA, C2A, CZ, E, C, D;
    do {
        SX = std::sin(X);
        CX = std::cos(X);
        TU1 = CU2 * SX;
        TU2 = BAZ - SU1 * CU2 * CX;
        SY = std::sqrt(TU1 * TU1 + TU2 * TU2);
        CY = S * CX + FAZ;
        Y = std::atan2(SY, CY);
        SA = S * SX / SY;
        C2A = -SA * SA + 1.;
        CZ = FAZ + FAZ;
        if (C2A > 0.) {
            CZ = -CZ / C2A + CY;
        }
        E = CZ * CZ * 2. - 1.;
        C = ((-3. * C2A + 4.) * F + 4.) * C2A * F / 16.;
        D = X;
        X = ((E * CY * C + CZ) * SY * C + Y) * SA;
        X = (1. - C) * X * F + lon2 - lon1;
    } while (std::abs(D - X) > (0.01));

    FAZ = std::atan2(TU1, TU2);
    BAZ = std::atan2(CU1 * SX, BAZ * CX - SU1 * CU2) + Constants::PI;
    X = std::sqrt((1. / R / R - 1.) * C2A + 1.) + 1.;
    X = (X - 2.) / X;
    C = 1. - X;
    C = (X * X / 4. + 1.) / C;
    D = (0.375 * X * X - 1.) * X;
    X = E * CY;
    S = 1. - E - E;
    S = ((((SY * SY * 4. - 3.) * S * CZ * D / 6. - X) * D / 4. + CZ) * SY * D + Y) * C * WGS84::a * R;

    output->distance = S;
    output->heading1 = FAZ * Constants::RTOD;
    output->heading2 = BAZ * Constants::RTOD;

    while (output->heading1 < 0) output->heading1 += 360;
    while (output->heading2 < 0) output->heading2 += 360;

    return output;
}

}  // namespace snapengine
}  // namespace alus
