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

#include <cstddef>
#include <vector>

#include "delaunay_triangle2D.h"

namespace alus::delaunay {

/**
 * TODO: DO NOT FORGET TO SORT OUT ALL INVALID INDEXES BEFORE ACTIVATING THIS CLASS!
 */
class DelaunayTriangulator {
private:
public:
    std::vector<DelaunayTriangle2D> host_triangles_;
    DelaunayTriangle2D* device_triangles_ = nullptr;
    size_t triangle_count_;  // yes you can use host_triangles.size() instead, but you can't do it on the gpu

    DelaunayTriangulator() = default;
    ~DelaunayTriangulator() = default;

    void TriangulateCPU(const double* x_coords, double x_multiplier, const double* y_coords, double y_multiplier,
                        int size, double invalid_index);
    void TriangulateCPU2(const double* x_coords, double x_multiplier, const double* y_coords, double y_multiplier,
                         int size, double invalid_index);
};

}  // namespace alus::delaunay