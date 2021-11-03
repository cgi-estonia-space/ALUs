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

#include "kernel_array.h"
#include "orbit_state_vector_computation.h"

namespace alus {
namespace terraincorrection {

/**
 * Representation of data that is needed for calculations only.
 *
 * Stripped down version of RangeDopplerTerrainMetadata with only POD types and without unnecessary fields.
 */
struct ComputationMetadata {
    double first_line_time_mjd;
    double last_line_time_mjd;
    double first_near_lat;
    double first_near_long;
    double first_far_lat;
    double first_far_long;
    double last_near_lat;
    double last_near_long;
    double last_far_lat;
    double last_far_long;
    double radar_frequency;
    double range_spacing;
    double line_time_interval;
    double avg_scene_height;
    double slant_range_to_first_pixel;
    int first_valid_pixel;
    int last_valid_pixel;
    double first_valid_line_time;
    double last_valid_line_time;
};

}  // namespace terraincorrection
}  // namespace alus