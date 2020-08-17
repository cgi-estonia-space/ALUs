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

#include <pos_vector.h>
#include <cmath>

#include "orbit_state_vector.h"
#include "pos_vector.h"

namespace alus {
namespace s1tbx {
namespace orbitstatevectors {

constexpr int NV{8}; //TODO: does anyone know what this is supposed to be?


/**
 *  Important note, timeMap in the original code is just a cache to make things faster. We can't afford this on the gpu.
 * @param time          Somesort of time. No idea which one
 * @param orbit         An array of OrbitStateVectors
 * @param numOrbitVec   Length of OrbitStateVectors Array
 * @param dt            OrbitStateVectors.dt variable.
 * @param position      position of something, perhaps the satelite? Will be filled
 * @param velocity      velocity of something, üerhaps the satelite? Makes up the lagrange Interpolating Polynomial with position. Will be filled.
 */
inline __device__ __host__ void getPositionVelocity(double time,
                                                    snapengine::OrbitStateVector *orbit,
                                                    const int numOrbitVec,
                                                    const double dt,
                                                    snapengine::PosVector *position,
                                                    snapengine::PosVector *velocity) {

    int i0, iN;
    if (numOrbitVec <= NV) {
        i0 = 0;
        iN = numOrbitVec - 1;
    } else {
        i0 = max((int) ((time - orbit[0].timeMjd_) / dt) - NV / 2 + 1, 0);
        iN = min(i0 + NV - 1, numOrbitVec - 1);
        i0 = (iN < numOrbitVec - 1 ? i0 : iN - NV + 1);
    }

    for (int i = i0; i <= iN; ++i) {
        snapengine::OrbitStateVector orbI = orbit[i];

        double weight = 1;
        for (int j = i0; j <= iN; ++j) {
            if (j != i) {
                const double time2 = orbit[j].timeMjd_;
                weight *= (time - time2) / (orbI.timeMjd_ - time2);
            }
        }

        position->x += weight * orbI.xPos_;
        position->y += weight * orbI.yPos_;
        position->z += weight * orbI.zPos_;

        velocity->x += weight * orbI.xVel_;
        velocity->y += weight * orbI.yVel_;
        velocity->z += weight * orbI.zVel_;
    }
}

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
