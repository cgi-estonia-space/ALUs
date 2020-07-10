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

#include "GeoUtils.hpp"
#include "pos_vector.h"
#include "general_constants.h"


namespace alus {
namespace snapengine {
namespace geoutils {
/**
 * Convert geodetic coordinate into cartesian XYZ coordinate with specified geodetic system.
 *
 * Duplicate of a SNAP's geoutils.java's geo2xyzWGS84() for native.
 * Copied from a snap-engine's(https://github.com/senbox-org/snap-engine) repository originally stated to be implemented
 * by "Copyright (C) 2014 by Array Systems Computing Inc. http://www.array.ca"
 *
 * @param latitude  The latitude of a given pixel (in degree).
 * @param longitude The longitude of the given pixel (in degree).
 * @param altitude  The altitude of the given pixel (in m)
 * @param xyz       The xyz coordinates of the given pixel.
 */
inline __device__ __host__ void Geo2xyzWgs84Impl(double latitude, double longitude, double altitude, PosVector& xyz) {
    double const lat = latitude * constants::DTOR;
    double const lon = longitude * constants::DTOR;

    double const sinLat = sin(lat);
    double const N = (WGS84::a / sqrt(1.0 - WGS84::e2 * sinLat * sinLat));
    double const NcosLat = (N + altitude) * cos(lat);

    xyz.x = NcosLat * cos(lon);  // in m
    xyz.y = NcosLat * sin(lon);  // in m
    xyz.z = (N + altitude - WGS84::e2 * N) * sinLat;
    // xyz.z = (WGS84.e2inv * N  + altitude) * sinLat;
}
}  // namespace geoutils
}  // namespace snapEngine
}  // namespace alus