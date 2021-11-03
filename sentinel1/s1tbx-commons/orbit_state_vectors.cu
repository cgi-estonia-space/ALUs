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

#include "../../snap-engine/pos_vector.h"
#include "cuda_util.cuh"
#include "s1tbx-commons/orbit_state_vectors.cuh"

namespace alus {
namespace s1tbx {
namespace orbitstatevectors {
snapengine::PosVector GetPosition(double time, cuda::KernelArray<snapengine::OrbitStateVectorComputation> vectors) {
    return GetPositionImpl(time, vectors);
}
}  // namespace orbitstatevectors
}  // namespace s1tbx
}  // namespace alus