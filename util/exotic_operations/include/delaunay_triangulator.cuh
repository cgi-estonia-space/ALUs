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

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "delaunay_triangle2D.h"

namespace alus {
namespace delaunay {

cudaError_t LaunchDelaunayTriangulation(double *x_coords, double *y_coords, int width, int height, DelaunayTriangle2Dgpu *triangles);

}//namespace
}//namespace
