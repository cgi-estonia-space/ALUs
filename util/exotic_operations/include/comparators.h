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

#include <iostream>

#include "delaunay_triangle2D.h"

namespace alus {

size_t EqualsArrays(const float *a, const float *b, int elems);
size_t EqualsArrays(const float *a, const float *b, int elems, float delta);
size_t EqualsArraysd(const double *a, const double *b, int elems);
size_t EqualsArraysd(const double *a, const double *b, int elems, double delta);
size_t EqualsArrays2Dd(const double *const *a, const double *const *b, int x, int y);

int EqualsDouble(double a, double b, double delta);
int EqualsTrianglesByIndices(delaunay::DelaunayTriangle2D a, delaunay::DelaunayTriangle2D b);
int EqualsTrianglesByPoints(delaunay::DelaunayTriangle2D a, delaunay::DelaunayTriangle2D b, double delta);
size_t EqualsTriangles(delaunay::DelaunayTriangle2D *a, delaunay::DelaunayTriangle2D *b, size_t length, double delta);
}  // namespace alus
