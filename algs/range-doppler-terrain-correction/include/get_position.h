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

#include "kernel_array.h"
#include "orbit_state_vector_computation.h"
#include "pos_vector.h"
#include "position_data.h"

namespace alus {
namespace terraincorrection {

struct GetPositionMetadata {
    double first_line_utc;
    double line_time_interval;
    double wavelength;
    double range_spacing;
    double near_edge_slant_range;
    int source_image_width;
    cuda::KernelArray<snapengine::PosVector> sensor_position;
    cuda::KernelArray<snapengine::PosVector> sensor_velocity;
    cuda::KernelArray<snapengine::OrbitStateVectorComputation> orbit_state_vectors;
    cuda::KernelArray<double> orbit_state_vector_lut;
};

/**
 * Calculates a lookup table for the orbit state interpolation, this can be used to turn a divide into a multiply
 * @param[in] orbit_state_vector
 * @return 2d array with precalculated coefficients
 */
inline std::vector<double> CalculateOrbitStateVectorLUT(
    const std::vector<alus::snapengine::OrbitStateVectorComputation>& comp_orbits) {
    const auto& osv = comp_orbits;
    std::vector<double> h_lut;
    for (size_t i = 0; i < osv.size(); i++) {
        for (size_t j = 0; j < osv.size(); j++) {
            double timei = osv[i].timeMjd_;
            double timej = osv[j].timeMjd_;
            if (timei != timej) {
                h_lut.push_back(1 / (timei - timej));
            } else {
                h_lut.push_back(0);
            }
        }
    }
    return h_lut;
}

}  // namespace terraincorrection
}  // namespace alus
