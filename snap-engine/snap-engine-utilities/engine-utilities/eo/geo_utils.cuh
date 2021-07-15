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

#include "../../../geo_utils.h"
#include "../../../pos_vector.h"
#include "general_constants.h"
#include "snap-core/core/util/geo_utils.h"
#include "snap-engine-utilities/engine-utilities/eo/constants.h"

namespace alus {
namespace snapengine {
namespace geoutils {
/**
 * Convert geodetic coordinate into cartesian XYZ coordinate with specified geodetic system.
 *
 * Duplicate of a SNAP's geoutils.java's geo2xyzWGS84() for native.
 * Copied from (https://github.com/senbox-org/snap-engine). It was originally stated:
 * by "Copyright (C) 2014 by Array Systems Computing Inc. http://www.array.ca"
 *
 * @param latitude  The latitude of a given pixel (in degree).
 * @param longitude The longitude of the given pixel (in degree).
 * @param altitude  The altitude of the given pixel (in m)
 * @param xyz       The xyz coordinates of the given pixel.
 */

inline __device__ __host__ void Geo2xyzWgs84Impl(double latitude, double longitude, double altitude, PosVector& xyz) {
    double const lat = latitude * eo::constants::DTOR;
    double const lon = longitude * eo::constants::DTOR;

    double sin_lat;
    double cos_lat;
    sincos(lat, &sin_lat, &cos_lat);

    double const sinLat = sin_lat;

    double const N = (snapengine::WGS84::A / sqrt(1.0 - snapengine::WGS84::E2 * sinLat * sinLat));
    double const NcosLat = (N + altitude) * cos_lat;

    double sin_lon;
    double cos_lon;
    sincos(lon, &sin_lon, &cos_lon);

    xyz.x = NcosLat * cos_lon;  // in m
    xyz.y = NcosLat * sin_lon;  // in m
    xyz.z = (N + altitude - snapengine::WGS84::E2 * N) * sinLat;
    // xyz.z = (WGS84.e2inv * N  + altitude) * sinLat;
}
}  // namespace geoutils
}  // namespace snapengine
}  // namespace alus