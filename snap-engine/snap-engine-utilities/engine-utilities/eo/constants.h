/**
 * This file is a filtered duplicate of a SNAP's
 * org.esa.snap.engine_utilities.eo.Constants.java
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

#include "snap-core/core/util/geo_utils.h"

namespace alus {
namespace snapengine {
namespace eo {
namespace constants {
constexpr double SECONDS_IN_DAY = 86400.0;
constexpr double LIGHT_SPEED = 299792458.0;  //  m / s
constexpr double HALF_LIGHT_SPEED = LIGHT_SPEED / 2.0;
constexpr double LIGHT_SPEED_IN_METERS_PER_DAY = LIGHT_SPEED * SECONDS_IN_DAY;

// todo::add if needed
constexpr double SEMI_MAJOR_AXIS = WGS84::A;  // in m, WGS84 semi-major axis of Earth
constexpr double SEMI_MINOR_AXIS = WGS84::B;  // in m, WGS84 semi-minor axis of Earth

constexpr double MEAN_EARTH_RADIUS = 6371008.7714;  // in m (WGS84)

constexpr double ONE_MILLION = 1000000.0;
constexpr double TEN_MILLION = 10000000.0;
constexpr double ONE_BILLION = 1000000000.0;
constexpr double ONE_BILLIONTH = 1.0 / ONE_BILLION;

constexpr double PI =
    3.1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348;
constexpr double HALF_PI = PI * 0.5;
constexpr double TWO_PI = 2.0 * PI;
// todo::add if needed
// static constexpr double SQRT2 = std::sqrt(2);

constexpr double DTOR = PI / 180.0;
constexpr double RTOD = 180.0 / PI;

constexpr double EPS = 1e-15;

constexpr double NO_DATA_VALUE = -99999.0;

constexpr double S_TO_NS = ONE_BILLION;    // s to ns
constexpr double NS_TO_S = ONE_BILLIONTH;  // ns to s
}  // namespace constants
}  // namespace eo
}  // namespace snapengine
}  // namespace alus
