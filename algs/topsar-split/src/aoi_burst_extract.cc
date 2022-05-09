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

#include "aoi_burst_extract.h"

#include <vector>

#include <boost/geometry.hpp>

#include "product.h"
#include "snap-core/core/datamodel/tie_point_grid.h"
#include "operator_utils.h"
#include "topsar_split.h"

#include "alus_log.h"

namespace alus::topsarsplit {
BurstBox GetBurstBoxFrom(const std::vector<alus::Coordinates>& upper_burst_edge,
                         const std::vector<alus::Coordinates>& bottom_burst_edge) {
    const auto last_coordinate_index = upper_burst_edge.size() - 1;
    BurstBox burst_boundaries;
    // Construct box like polygon out of the edges' min/max points.
    const Point polygon_closing_point(upper_burst_edge.at(0).lon, upper_burst_edge.at(0).lat);
    burst_boundaries.outer().push_back(polygon_closing_point);
    burst_boundaries.outer().push_back(
        Point(upper_burst_edge.at(last_coordinate_index).lon, upper_burst_edge.at(last_coordinate_index).lat));
    burst_boundaries.outer().push_back(
        Point(bottom_burst_edge.at(last_coordinate_index).lon, bottom_burst_edge.at(last_coordinate_index).lat));
    burst_boundaries.outer().push_back(Point(bottom_burst_edge.at(0).lon, bottom_burst_edge.at(0).lat));
    // Close the polygon
    burst_boundaries.outer().push_back(polygon_closing_point);

    return burst_boundaries;
}

bool IsCovered(const BurstBox& burst, const Aoi& aoi) {
    std::vector<Point> points;
    boost::geometry::intersection(aoi, burst, points);
    if (!points.empty()) {
        return true;
    }

    return boost::geometry::within(aoi, burst) || boost::geometry::within(burst, aoi);
}

std::vector<int> DetermineBurstIndexesCoveredBy(const Aoi& aoi,
                                                const std::vector<std::vector<Coordinates>>& burst_edge_line_points) {
    std::vector<int> selected_bursts{};
    for (size_t i = TopsarSplit::BURST_INDEX_OFFSET; i < burst_edge_line_points.size(); ++i) {
        const auto& upper_line = burst_edge_line_points.at(i - TopsarSplit::BURST_INDEX_OFFSET);
        const auto& bottom_line = burst_edge_line_points.at(i);
        if (IsCovered(GetBurstBoxFrom(upper_line, bottom_line), aoi)) {
            selected_bursts.push_back(static_cast<int>(i));
        }
    }

    return selected_bursts;
}

SwathPolygon ExtractSwathPolygon(std::shared_ptr<snapengine::Product> product) {

    auto tpg_lat = product->GetTiePointGrid(snapengine::OperatorUtils::TPG_LATITUDE);
    auto tpg_lon = product->GetTiePointGrid(snapengine::OperatorUtils::TPG_LONGITUDE);
    alus::s1tbx::Sentinel1Utils slave_utils(product);
    alus::s1tbx::SubSwathInfo* subswath = slave_utils.subswath_.at(0).get();

    std::vector<Point> left;
    std::vector<Point> right;

    for (int i = 0; i < subswath->num_of_bursts_; i++) {
        const int first_line = subswath->first_valid_line_.at(i);
        const int last_line = subswath->last_valid_line_.at(i);
        const int y_burst_start = i * subswath->lines_per_burst_;
        const int y_burst_end = y_burst_start + subswath->lines_per_burst_;
        const int x_valid_start = subswath->first_valid_sample_.at(i).at(first_line);
        const int x_valid_end = subswath->last_valid_sample_.at(i).at(last_line);

        left.push_back({tpg_lon->GetPixelDouble(x_valid_start, y_burst_start),
                        tpg_lat->GetPixelDouble(x_valid_start, y_burst_start)});
        left.push_back(
            {tpg_lon->GetPixelDouble(x_valid_start, y_burst_end), tpg_lat->GetPixelDouble(x_valid_start, y_burst_end)});

        right.push_back(
            {tpg_lon->GetPixelDouble(x_valid_end, y_burst_start), tpg_lat->GetPixelDouble(x_valid_end, y_burst_start)});
        right.push_back(
            {tpg_lon->GetPixelDouble(x_valid_end, y_burst_end), tpg_lat->GetPixelDouble(x_valid_end, y_burst_end)});
    }

    BurstBox polygon;
    for (const auto& point : left) {
        polygon.outer().push_back(point);
    }
    std::reverse(right.begin(), right.end());
    for (const auto& point : right) {
        polygon.outer().push_back(point);
    }

    LOGD << "Swath = " << subswath->subswath_name_ << " boundary = " << boost::geometry::wkt(polygon);
    return polygon;
}

bool IsWithinSwath(const Aoi& aoi, const SwathPolygon& swath_polygon) {
    return boost::geometry::within(aoi, swath_polygon);
}

}  // namespace alus::topsarsplit