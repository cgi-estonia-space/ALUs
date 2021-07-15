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

#include "delaunay_triangle2D.h"

namespace alus {
namespace snapengine {
namespace triangularinterpolation {

/**
 * A substantial simplification for org.jlinda.core.Window from s1tbx.
 * We just don't need all of that stuff on the gpu.
 */
struct Window {
    /**
     * min line coordinate
     */
    long linelo;

    /**
     * max line coordinate
     */
    long linehi;

    /**
     * min pix coordinate
     */
    long pixlo;

    /**
     * max pix coordinate
     */
    long pixhi;
    /**
     * usually linehi - linelo + 1
     */
    int lines;
    /**
     * usually pixhi - pixlo + 1
     */
    int pixels;
};

/**
 * A port from org.jlinda.core.delaunay.TriangleInterpolator, which has a class with te same name.
 * NB both input and output arrays are 2d arrays which are written column by column rather than the usual line by line.
 */
struct Zdata {
    double *input_arr;
    size_t input_width, input_height;
    double *output_arr;
    size_t output_width, output_height;
    int min_int;
    int max_int;
};

struct Zdataabc {
    double a, b, c;
};

struct PointInTriangle {
    double xtd0, xtd1, xtd2, ytd0, ytd1, ytd2;
};

struct TriangleInterpolationParams {
    size_t triangle_count;
    size_t z_data_count;
    double xy_ratio;
    double x_scale;
    double y_scale;
    double offset;
    double invalid_index;
    Window window;
};

struct InterpolationParams{
    size_t z_data_count;
    double x_scale;
    double y_scale;
    double offset;
    size_t accepted_triangles;
    Window window;
};

struct TriangleDto {
    PointInTriangle point_in_triangle;
    long i_min;
    long i_max;
    long j_min;
    long j_max;
    double xp;
    double yp;
    double vx[3];
    double vy[3];
};

cudaError_t LaunchInterpolation(delaunay::DelaunayTriangle2D *triangles, Zdata* zdata,
                                TriangleInterpolationParams params);

}  // namespace triangularinterpolation
}  // namespace snapengine
}  // namespace alus
