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

#include <cmath>

#include "pos_vector.h"
#include "orbit_state_vector.h"

namespace alus {
namespace s1tbx {
namespace orbitstatevectors {

inline __device__ __host__ snapengine::PosVector GetPositionImpl(
    double time, cudautil::KernelArray<snapengine::OrbitStateVector> vectors) {
    const int nv{8};
    const int vectorsSize = vectors.size;
    // TODO: This should be done once.
    const double dt =
        (vectors.array[vectorsSize - 1].timeMjd_ - vectors.array[0].timeMjd_) / static_cast<double>(vectorsSize - 1);

    int i0;
    int iN;
    if (vectorsSize <= nv) {
        i0 = 0;
        iN = static_cast<int>(vectorsSize - 1);
    } else {
        i0 = std::max((int)((time - vectors.array[0].timeMjd_) / dt) - nv / 2 + 1, 0);
        iN = std::min(i0 + nv - 1, vectorsSize - 1);
        i0 = (iN < vectorsSize - 1 ? i0 : iN - nv + 1);
    }

    snapengine::PosVector result{0, 0, 0};
    for (int i = i0; i <= iN; ++i) {
        auto const orbI = vectors.array[i];

        double weight = 1;
        for (int j = i0; j <= iN; ++j) {
            if (j != i) {
                double const time2 = vectors.array[j].timeMjd_;
                weight *= (time - time2) / (orbI.timeMjd_ - time2);
            }
        }
        result.x += weight * orbI.xPos_;
        result.y += weight * orbI.yPos_;
        result.z += weight * orbI.zPos_;
    }
    return result;
}

}  // namespace orbitstatevectors
}  // namespace s1tbx
}  // namespace alus
