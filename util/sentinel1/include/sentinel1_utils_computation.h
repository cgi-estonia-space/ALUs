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

namespace alus {
namespace s1tbx {
struct DeviceSentinel1Utils {
    double first_line_utc;
    double last_line_utc;
    double line_time_interval;
    double near_edge_slant_range;
    double wavelength;
    double range_spacing;
    double azimuth_spacing;

    int source_image_width;
    int source_image_height;
    int near_range_on_left;
    int srgr_flag;
};

struct Sentinel1Index {
    int i0;
    int i1;
    int j0;
    int j1;
    double mu_x;
    double mu_y;
};
}
}