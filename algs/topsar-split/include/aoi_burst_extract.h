/**
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

#include <vector>

#include <boost/geometry.hpp>

#include "product.h"
#include "raster_properties.h"


namespace alus::topsarsplit {

// Definition of point as a boost::geometry type
using Point = boost::geometry::model::d2::point_xy<Coordinate>;
// Burst box geometry is defined by a polygon of points
using BurstBox = boost::geometry::model::polygon<Point>;
using SwathPolygon = boost::geometry::model::polygon<Point>;
// Area of interest is defined by a polygon of points
using Aoi = boost::geometry::model::polygon<Point>;
BurstBox GetBurstBoxFrom(const std::vector<alus::Coordinates>& upper_burst_edge,
                         const std::vector<alus::Coordinates>& bottom_burst_edge);
bool IsCovered(const BurstBox& burst, const Aoi& aoi);
std::vector<int> DetermineBurstIndexesCoveredBy(const Aoi& aoi,
                                                const std::vector<std::vector<Coordinates>>& burst_edge_line_points);
bool IsWithinSwath(const Aoi& aoi, std::shared_ptr<snapengine::Product> product);
bool IsWithinSwath(const Aoi& aoi, const SwathPolygon& swath_polygon);
SwathPolygon ExtractSwathPolygon(std::shared_ptr<snapengine::Product> product);

}  // namespace alus::topsarsplit