/**
 * This file is a filtered duplicate of a SNAP's
 * org.esa.s1tbx.commons.OrbitStateVectors.java
 * ported for native code.
 * Copied from (https://github.com/senbox-org/snap-engine). It was originally stated:
 *
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

#include <memory>
#include <unordered_map>
#include <vector>

#include "kernel_array.h"
#include "orbit_state_vector.h"
#include "orbit_state_vector_computation.h"
#include "pos_vector.h"

namespace alus::s1tbx {
class OrbitStateVectors {
public:
    class PositionVelocity;  // forward declaration
    std::vector<snapengine::OrbitStateVector> orbit_state_vectors_;
    std::vector<snapengine::OrbitStateVectorComputation> orbit_state_vectors_computation_;
    std::vector<snapengine::PosVector> sensor_position_;
    std::vector<snapengine::PosVector> sensor_velocity_;

    OrbitStateVectors(std::vector<snapengine::OrbitStateVector> orbit_state_vectors, double first_line_utc,
                      double line_time_interval, int source_image_height);
    explicit OrbitStateVectors(std::vector<snapengine::OrbitStateVector> orbit_state_vectors);
    std::shared_ptr<PositionVelocity> GetPositionVelocity(double time);
    std::unique_ptr<snapengine::PosVector> GetPosition(double time, std::unique_ptr<snapengine::PosVector> position);
    std::unique_ptr<snapengine::PosVector> GetVelocity(double time);
    double GetDt() const { return dt_; }

private:
    double dt_ = 0.0;
    std::unordered_map<double, std::shared_ptr<PositionVelocity>> time_map_;

    static constexpr int NV{8};
    static std::vector<snapengine::OrbitStateVector> RemoveRedundantVectors(
        std::vector<snapengine::OrbitStateVector> orbit_state_vectors);
};

namespace orbitstatevectors {

/**
 * Ported function from OrbitStateVectors::GetPosition() off of S1TBX repo.
 *
 * Position argument is missing, because originally it was modified in place, but this is returning that value here.
 * This is compiled on CUDA, therefore available for kernel code too.
 *
 * @param time Zero point doppler time.
 * @param vectors Supplied here manually, in original implementation they are a member field of Orbit class.
 * @return snapengine::PosVector Position vector calculated
 */
snapengine::PosVector GetPosition(double time, cuda::KernelArray<snapengine::OrbitStateVectorComputation> vectors);

}  // namespace orbitstatevectors
}  // namespace alus::s1tbx