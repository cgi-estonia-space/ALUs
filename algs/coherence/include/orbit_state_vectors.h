/**
 * This file is a filtered duplicate of a SNAP's org.esa.s1tbx.commons.OrbitStateVectors.java
 * ported for native code.
 * Copied from a snap-engine's(https://github.com/senbox-org/snap-engine) repository originally stated
 * to be implemented by "Copyright (C) 2010 Brockmann Consult GmbH (info@brockmann-consult.de)"
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

#include "../../../util/snap-engine/include/pos_vector.h"
#include "orbit_state_vector.h"

namespace alus {
namespace snapengine {
class OrbitStateVectors {
   public:
    class PositionVelocity;  // forward declaration
    std::vector<OrbitStateVector> orbit_state_vectors_;
    std::vector<PosVector> sensor_position_;
    std::vector<PosVector> sensor_velocity_;

    OrbitStateVectors(std::vector<OrbitStateVector> orbit_state_vectors,
                      double first_line_Utc,
                      double line_time_interval,
                      int source_image_height);
    explicit OrbitStateVectors(std::vector<OrbitStateVector> orbit_state_vectors);
    std::shared_ptr<PositionVelocity> GetPositionVelocity(double time);
    std::unique_ptr<PosVector> GetPosition(double time, std::unique_ptr<PosVector> position);
    std::unique_ptr<PosVector> GetVelocity(double time);

   private:
    double dt_ = 0.0;
    std::unordered_map<double, std::shared_ptr<PositionVelocity>> time_map_;
    static constexpr int NV_{8};

    static std::vector<OrbitStateVector> RemoveRedundantVectors(std::vector<OrbitStateVector> orbit_state_vectors);
};

}  // namespace snapengine
}  // namespace alus
