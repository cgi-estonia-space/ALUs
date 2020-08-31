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

#include <cmath>
#include <vector>

#include "shapes.h"
#include "snap_triangle.h"
#include "delaunay_triangle2D.h"

namespace external {
namespace delaunay {


/**
 * This is meant to be a copy of org.jlinda.core.delaunay.FastDelaunayTriangulator.
 */
class SnapDelaunayTriangulator {
   private:
    /**
     * Fictive Coordinate representing the Horizon, or an infinite point.
     * It closes triangles around the convex hull of the triangulation
     */
    alus::PointDouble HORIZON;
    SnapTriangle *current_external_triangle_;
    std::vector<SnapTriangle*> triangles;


    void InitTriangulation(alus::PointDouble c0, alus::PointDouble c1);
    void AddExternalVertex(alus::PointDouble point);
    std::vector<SnapTriangle*> BuildTrianglesBetweenNewVertexAndConvexHull(alus::PointDouble point);
    void Link(SnapTriangle *t1, int side1, SnapTriangle *t2, int side2);
    void Link(SnapTriangle *t1, int side1, SnapTriangle *t2);
    void LinkExteriorTriangles(SnapTriangle *t1, SnapTriangle *t2);
    void Delaunay(SnapTriangle *t, int side);
    void Flip(SnapTriangle *t0, int side0, SnapTriangle *t1, int side1);

   public:
    SnapDelaunayTriangulator();
    ~SnapDelaunayTriangulator();
    void Triangulate(alus::PointDouble *p, int size);
    size_t TrianglesSize(){
        return triangles.size();
    }
    std::vector<alus::delaunay::DelaunayTriangle2D> Get2dTriangles();


};

}//namespace
}//namespace

