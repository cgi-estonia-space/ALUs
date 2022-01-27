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
#include "../../../snap-engine/orbit_state_vector_computation.h"
#include "../../../snap-engine/pos_vector.h"

namespace alus {
namespace tests {

struct ZeroDopplerTimeData {
    int data_size;
    double* device_line_time_interval;
    double* device_wavelengths;
    alus::snapengine::PosVector* device_earth_points;
    alus::snapengine::OrbitStateVectorComputation* orbit;
    int num_orbit_vec;
    double dt;
};

cudaError_t LaunchZeroDopplerTimeTest(dim3 grid_size, dim3 block_size, double* results, ZeroDopplerTimeData data);

}  // namespace tests
}  // namespace alus