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

#include "snap-core/util/geo_utils.h"

namespace alus {
namespace snapengine {
class Constants {
public:
    static constexpr double SECONDS_IN_DAY = 86400.0;
    static constexpr double LIGHT_SPEED = 299792458.0;  //  m / s
    static constexpr double HALF_LIGHT_SPEED = LIGHT_SPEED / 2.0;
    static constexpr double LIGHT_SPEED_IN_METERS_PER_DAY = LIGHT_SPEED * SECONDS_IN_DAY;

    // todo::add if needed
    static constexpr double SEMI_MAJOR_AXIS = WGS84::a; // in m, WGS84 semi-major axis of Earth
    static constexpr double SEMI_MINOR_AXIS = WGS84::b; // in m, WGS84 semi-minor axis of Earth

    static constexpr double MEAN_EARTH_RADIUS = 6371008.7714;  // in m (WGS84)

    static constexpr double ONE_MILLION = 1000000.0;
    static constexpr double TEN_MILLION = 10000000.0;
    static constexpr double ONE_BILLION = 1000000000.0;
    static constexpr double ONE_BILLIONTH = 1.0 / ONE_BILLION;

    static constexpr double PI = 3.14159265358979323846264338327950288;
    static constexpr double _PI =
        3.1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348;
    static constexpr double TWO_PI = 2.0 * PI;
    static constexpr double HALF_PI = PI * 0.5;
    static constexpr double _TWO_PI = 2.0 * _PI;
    // todo::add if needed
    // static constexpr double SQRT2 = std::sqrt(2);

    static constexpr double DTOR = PI / 180.0;
    static constexpr double RTOD = 180.0 / PI;

    static constexpr double _DTOR = _PI / 180.0;
    static constexpr double _RTOD = 180.0 / _PI;

    static constexpr double EPS = 1e-15;

    static constexpr double NO_DATA_VALUE = -99999.0;

    static constexpr double STONS = ONE_BILLION;    // s to ns
    static constexpr double NSTOS = ONE_BILLIONTH;  // ns to s
};
}  // namespace snapengine
}  // namespace alus
