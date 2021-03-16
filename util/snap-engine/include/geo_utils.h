/**
 * This file is a filtered duplicate of a SNAP's
 * geoutils.java
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

#include "pos_vector.h"

namespace alus {
namespace snapengine {
namespace geoutils {
/**
 * Duplicate of a SNAP's geoutils.java's geo2xyzWGS84().
 *
 * This actually exists as an inline version for CUDA calls as Geo2xyzWgs84Impl() in geoutils.cuh.
 * This procedure is duplicated by the nvcc for host processing in geoutils.cu.
 */
void Geo2xyzWgs84(double latitude, double longitude, double altitude, PosVector& xyz);

namespace WGS84 {
constexpr double a{6378137.0};                          // m
constexpr double b{6356752.3142451794975639665996337};  // 6356752.31424518; // m
constexpr double earthFlatCoef{1.0 / ((a - b) / a)};    // 298.257223563;
constexpr double e2{2.0 / earthFlatCoef - 1.0 / (earthFlatCoef * earthFlatCoef)};
constexpr double e2inv{1 - e2};
constexpr double ep2{e2 / (1 - e2)};
}  // namespace WGS84
}  // namespace geoutils
}  // namespace snapEngine
}  // namespace alus
