/*
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
#include "jlinda/jlinda-core/geopoint.h"

#include <cmath>
#include <limits>
#include <sstream>
#include <string>

namespace alus {
namespace jlinda {

std::string GeoPoint::GetDegreeString(double value, bool longitudial) {
    int sign = (value == 0.0) ? 0 : (value < 0.0) ? -1 : 1;
    double rest = std::abs(value);
    int degree = FloorInt(rest);
    rest -= degree;
    int minutes = FloorInt(_MIN_PER_DEG * rest);
    rest -= minutes / _MIN_PER_DEG;
    double seconds = (_SEC_PER_DEG * rest);
    rest -= seconds / _SEC_PER_DEG;
    if (seconds == 60) {
        seconds = 0;
        minutes++;
        if (minutes == 60) {
            minutes = 0;
            degree++;
        }
    }

    std::stringstream ss;
    ss << degree;
    ss << '\260';  // degree
    if (minutes != 0 || seconds != 0) {
        if (minutes < 10) {
            ss << '0';
        }
        ss << minutes;
        ss << '\'';
        if (seconds != 0) {
            if (seconds < 10) {
                ss << '0';
            }
            ss << seconds;
            ss << '"';
        }
    }
    if (sign == -1) {
        ss << ' ';
        if (longitudial) {
            ss << 'W';
        } else {
            ss << 'S';
        }
    } else if (sign == 1) {
        ss << ' ';
        if (longitudial) {
            ss << 'E';
        } else {
            ss << 'N';
        }
    }

    return ss.str();
}

bool GeoPoint::IsLatValid(double lat) { return lat >= -90.0 && lat <= 90.0; }

bool GeoPoint::IsLonValid(double lon) { return !std::isnan(lon) && !std::isinf(lon); }

int GeoPoint::FloorInt(const double value) { return static_cast<int>(floor(value)); }

GeoPoint::GeoPoint() = default;

GeoPoint::GeoPoint(const GeoPoint& geo_point) : GeoPoint(geo_point.lat_, geo_point.lon_) {}

GeoPoint::GeoPoint(double lat, double lon) : lat_{lat}, lon_{lon} {}
void GeoPoint::SetLocation(double lat, double lon) {
    lat_ = lat;
    lon_ = lon;
}

bool GeoPoint::IsValid() const { return IsLatValid(lat_) && IsLonValid(lon_); }

void GeoPoint::SetInvalid() {
    lat_ = std::numeric_limits<double>::quiet_NaN();
    lon_ = std::numeric_limits<double>::quiet_NaN();
}

std::string GeoPoint::ToString() const { return "[" + GetLatString() + "," + GetLonString() + "]"; }

void GeoPoint::Normalize() { lon_ = NormalizeLon(lon_); }

double GeoPoint::NormalizeLon(double lon) {
    if (lon < -360.0 || lon > 360.0) {
        lon = std::fmod(lon, 360.0);
    }
    if (lon < -180.0) {
        lon += 360.0;
    } else if (lon > 180.0) {
        lon -= 360.0;
    }
    return lon;
}

std::string GeoPoint::GetLatString() const { return GetLatString(lat_); }

std::string GeoPoint::GetLonString() const { return GetLonString(lon_); }

std::string GeoPoint::GetLatString(double lat) {
    if (IsLatValid(lat)) {
        return GetDegreeString(lat, false);
    }
    return "Inv N";
}

std::string GeoPoint::GetLonString(double lon) {
    if (IsLonValid(lon)) {
        return GetDegreeString(lon, true);
    }
    return "Inv E";
}

}  // namespace jlinda
}  // namespace alus
