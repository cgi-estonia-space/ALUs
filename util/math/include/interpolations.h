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

#ifdef __CUDACC__
#define DEVICE_STUB __device__
#define HOST_STUB __host__
#else
#define DEVICE_STUB
#define HOST_STUB
#endif

namespace alus::math::interpolations {

inline DEVICE_STUB HOST_STUB double Linear(double a, double b, double weight) {
    return (1.0 - weight) * a + weight * b;
}

}  // namespace alus::math::interpolations