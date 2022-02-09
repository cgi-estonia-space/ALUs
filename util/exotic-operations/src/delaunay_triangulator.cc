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
#include "delaunay_triangulator.h"

#include "delaunay_triangulator_cpu.h"
#include "snap_delaunay_triangulator.h"
#include "snap_triangle.h"

#include "cuda_util.h"

namespace alus::delaunay {

void DelaunayTriangulator::TriangulateCPU(const double* x_coords, double x_multiplier, const double* y_coords,
                                          double y_multiplier, int size, double invalid_index) {
    std::vector<external::delaunay::ITRIANGLE> v;
    const int end_size = size + 3;
    std::vector<alus::PointDouble> p;
    int ntri = 0;
    int i;
    DelaunayTriangle2D temp_triangle;

    p.reserve(end_size);
    for (i = 0; i < size; i++) {
        if (x_coords[i] != invalid_index && y_coords[i] != invalid_index) {
            alus::PointDouble temp_point = {x_coords[i] * x_multiplier, y_coords[i] * y_multiplier, i};
            p.push_back(temp_point);
        }
    }

    v.resize(3 * p.size());  // NOLINT
    qsort(p.data(), p.size(), sizeof(alus::PointDouble), external::delaunay::XYZCompare);
    p.resize(p.size() + 3);                               // NOLINT
    Triangulate(p.size() - 3, p.data(), v.data(), ntri);  // NOLINT

    this->host_triangles_.resize(ntri);
    this->triangle_count_ = ntri;

    for (i = 0; i < ntri; i++) {
        temp_triangle.ax = p.at(v.at(i).p1).x;
        temp_triangle.ay = p.at(v.at(i).p1).y;
        temp_triangle.a_index = p.at(v.at(i).p1).index;

        temp_triangle.bx = p.at(v.at(i).p2).x;
        temp_triangle.by = p.at(v.at(i).p2).y;
        temp_triangle.b_index = p.at(v.at(i).p2).index;

        temp_triangle.cx = p.at(v.at(i).p3).x;
        temp_triangle.cy = p.at(v.at(i).p3).y;
        temp_triangle.c_index = p.at(v.at(i).p3).index;

        this->host_triangles_.at(i) = temp_triangle;
    }
}

int SnapCompareTo(const void* v1, const void* v2) {
    alus::PointDouble* p1;
    alus::PointDouble* p2;

    p1 = reinterpret_cast<alus::PointDouble*>(const_cast<void*>(v1));
    p2 = reinterpret_cast<alus::PointDouble*>(const_cast<void*>(v2));

    if (p1->x < p2->x) {
        return -1;
    }
    if (p1->x > p2->x) {
        return 1;
    }
    if (p1->y < p2->y) {
        return -1;
    }
    return p1->y > p2->y ? 1 : 0;
}

void DelaunayTriangulator::TriangulateCPU2(const double* x_coords, double x_multiplier, const double* y_coords,
                                           double y_multiplier, int size, double invalid_index) {
    std::vector<alus::PointDouble> points;
    points.reserve(size);
    int i;

    for (i = 0; i < size; i++) {
        if (x_coords[i] != invalid_index && y_coords[i] != invalid_index) {
            alus::PointDouble temp_point = {x_coords[i] * x_multiplier, y_coords[i] * y_multiplier, i};
            points.push_back(temp_point);
        }
    }

    qsort(points.data(), points.size(), sizeof(alus::PointDouble), SnapCompareTo);
    external::delaunay::SnapDelaunayTriangulator triangulator;
    triangulator.Triangulate(points.data(), points.size());

    this->host_triangles_ = triangulator.Get2dTriangles();
    this->triangle_count_ = this->host_triangles_.size();
}
}  // namespace alus::delaunay
