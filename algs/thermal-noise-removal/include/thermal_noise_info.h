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

#include <vector>
#include <string>

#include "s1tbx-commons/noise_azimuth_vector.h"
#include "s1tbx-commons/noise_vector.h"
#include "time_maps.h"

namespace alus::tnr {
struct ThermalNoiseInfo {
    int lines_per_burst{0}; // Used only for SLC
    std::vector<s1tbx::NoiseAzimuthVector> noise_azimuth_vectors;
    std::vector<s1tbx::NoiseVector> noise_range_vectors;
    std::vector<s1tbx::NoiseVector> burst_to_range_vector_map; // Used only for SLC
    TimeMaps time_maps; // Used only for GRD
};
}  // namespace alus::tnr