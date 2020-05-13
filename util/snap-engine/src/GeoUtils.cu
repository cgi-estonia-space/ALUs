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
#include "GeoUtils.cuh"

namespace alus {
namespace snapengine {
namespace GeoUtils {
__host__ void geo2xyzWGS84(double latitude, double longitude, double altitude, PosVector& xyz) {
    geo2xyzWGS84_fast(latitude, longitude, altitude, xyz);
}

}  // namespace GeoUtils
}  // namespace snapEngine
}  // namespace alus