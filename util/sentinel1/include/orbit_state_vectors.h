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

#include <algorithm>
#include <vector>

#include "cuda_util.cuh"
#include "PosVector.hpp"
#include "orbit_state_vector.h"

namespace alus {
namespace s1tbx {

class OrbitStateVectors {
   public:
    std::vector<snapengine::OrbitStateVector> orbitStateVectors;
    snapengine::PosVector GetVelocity(double time);

    static int testVectors();

    OrbitStateVectors();
    explicit OrbitStateVectors(std::vector<snapengine::OrbitStateVector> const& orbitStateVectors);
    ~OrbitStateVectors() = default;

   private:
    std::vector<snapengine::PosVector> sensorPosition;  // sensor position for all range lines
    std::vector<snapengine::PosVector> sensorVelocity;  // sensor velocity for all range lines
    double dt = 0.0;
    const int nv = 8;

    static void getMockData();
    static std::vector<snapengine::OrbitStateVector> RemoveRedundantVectors(
        std::vector<snapengine::OrbitStateVector> orbitStateVectors);
};

namespace orbitstatevectors {

/**
 * Ported function from OrbitStateVectors::getPosition() off of S1TBX repo.
 *
 * Position argument is missing, because originally it was modified in place, but this is returning that value here.
 *
 * @param time Zero point doppler time.
 * @param vectors Supplied here manually, in original implementation they are a member field of Orbit class.
 * @return snapengine::PosVector Position vector calculated
 */
snapengine::PosVector GetPosition(double time,
                                  KernelArray<snapengine::OrbitStateVector> vectors);

}

}  // namespace s1tbx
}  // namespace alus
