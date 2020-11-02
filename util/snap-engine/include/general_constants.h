/**
 * This file is a duplicate of a SNAP's Constant.java ported for native code.
 * Copied from a snap-engine's(https://github.com/senbox-org/snap-engine) repository originally stated to be implemented
 * by "Copyright (C) 2014 by Array Systems Computing Inc. http://www.array.ca"
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

#include "geo_utils.h"

namespace alus {
namespace snapengine {
namespace constants {

constexpr double secondsInDay{86400.0};
constexpr double lightSpeed{299792458.0};  //  m / s
constexpr double halfLightSpeed{lightSpeed / 2.0};
constexpr double lightSpeedInMetersPerDay{lightSpeed * secondsInDay};

constexpr double semiMajorAxis{geoutils::WGS84::a};  // in m, WGS84 semi-major axis of Earth
constexpr double semiMinorAxis{geoutils::WGS84::b};  // in m, WGS84 semi-minor axis of Earth

constexpr double MeanEarthRadius{6371008.7714};  // in m (WGS84)

constexpr double oneMillion{1000000.0};
constexpr double tenMillion{10000000.0};
constexpr double oneBillion{1000000000.0};
constexpr double oneBillionth{1.0 / oneBillion};

constexpr double PI{3.14159265358979323846264338327950288};
constexpr double _PI{3.1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348};
constexpr double TWO_PI{2.0 * PI};
constexpr double HALF_PI{PI * 0.5};
constexpr double _TWO_PI{2.0 * _PI};
constexpr double sqrt2{1.41421356237};

constexpr double DTOR{PI / 180.0};
constexpr double RTOD{180.0 / PI};

constexpr double _DTOR{_PI / 180.0};
constexpr double _RTOD{180.0 / _PI};

constexpr double EPS{1e-15};

constexpr double NO_DATA_VALUE{-99999.0};

constexpr double sTOns{oneBillion};    // s to ns
constexpr double nsTOs{oneBillionth};  // ns to s

}  // namespace constants
}  // namespace snapEngine
}  // namespace alus