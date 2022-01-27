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

namespace alus {  // NOLINT
namespace delaunay {

/**
 * Is represented as y= mx + c.
 */
struct StraightEquation2D {
    double m;
    double c;
};

/**
 * This is to keep all the information needed about triangles to calculate the delaunay triangulation on the GPU.
 * The triangles also work as a makeshift linked list with each one knowing which was reported previously by its owner.
 */
struct DelaunayTriangle2Dgpu {
    double ax, ay, bx, by, cx, cy;
    int owner;                      // the point that created this triangle
    int previous;                   // the triangle that my owner created before me.
    int a_index, b_index, c_index;  // indexes to points in the point array
    StraightEquation2D bc_equation;
};

struct DelaunayTriangle2D {
    double ax, ay, bx, by, cx, cy;
    int a_index, b_index, c_index;  // snap logic requires those points traced.
};

}  // namespace delaunay
}  // namespace alus