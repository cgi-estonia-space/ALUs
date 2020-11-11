#include "geo_pos.h"

#include <cmath>
#include <limits>
#include <sstream>
#include <string>

namespace alus {
namespace snapengine {

std::string GeoPos::GetDegreeString(double value, bool longitudial, bool compass_format, bool decimal_format) {
    int sign = (value == 0.0F) ? 0 : (value < 0.0F) ? -1 : 1;
    double rest = std::abs(value);
    int degree = FloorInt(rest);
    rest -= degree;
    int minutes = FloorInt(MIN_PER_DEG * rest);
    rest -= minutes / MIN_PER_DEG;
    double seconds = (SEC_PER_DEG * rest);
    rest -= seconds / SEC_PER_DEG;
    if (seconds == 60) {
        seconds = 0;
        minutes++;
        if (minutes == 60) {
            minutes = 0;
            degree++;
        }
    }

    std::stringstream ss;

    if (!compass_format && sign == -1) {
        ss << "- ";
    }
    if (decimal_format) {
        ss << std::abs(value);
    } else {
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
    }

    if (compass_format) {
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
    }

    return ss.str();
}

bool GeoPos::IsLatValid(double lat) { return lat >= -90.0 && lat <= 90.0; }

bool GeoPos::IsLonValid(double lon) { return !std::isnan(lon) && !std::isinf(lon); }

int GeoPos::FloorInt(const double value) { return static_cast<int>(floor(value)); }

GeoPos::GeoPos() = default;

GeoPos::GeoPos(const GeoPos& geo_point) : GeoPos(geo_point.lat_, geo_point.lon_) {}

GeoPos::GeoPos(double lat, double lon) : lat_{lat}, lon_{lon} {}
void GeoPos::SetLocation(double lat, double lon) {
    lat_ = lat;
    lon_ = lon;
}

bool GeoPos::IsValid() const { return IsLatValid(lat_) && IsLonValid(lon_); }

void GeoPos::SetInvalid() {
    lat_ = std::numeric_limits<double>::quiet_NaN();
    lon_ = std::numeric_limits<double>::quiet_NaN();
}

std::string GeoPos::ToString() const { return "[" + GetLatString() + "," + GetLonString() + "]"; }

void GeoPos::Normalize() { lon_ = NormalizeLon(lon_); }

double GeoPos::NormalizeLon(double lon) {
    if (lon < -360.0f || lon > 360.0f) {
        lon = std::fmod(lon, 360.0f);
    }
    if (lon < -180.0f) {
        lon += 360.0f;
    } else if (lon > 180.0f) {
        lon -= 360.0f;
    }
    return lon;
}

std::string GeoPos::GetLatString() const { return GetLatString(lat_, true, false); }

std::string GeoPos::GetLatString(double lat, bool compass_format, bool decimal_format) {
    if (IsLatValid(lat)) {
        return GetDegreeString(lat, false, compass_format, decimal_format);
    }
    return "Inv N " + std::to_string(lat) + ")";
}
std::string GeoPos::GetLatString(double lat) { return GetLatString(lat, true, false); }

std::string GeoPos::GetLatString(bool compass_format, bool decimal_format) {
    return GetLatString(lat_, compass_format, decimal_format);
}

std::string GeoPos::GetLonString() const { return GetLonString(lon_, true, false); }

std::string GeoPos::GetLonString(double lon, bool compass_format, bool decimal_format) {
    if (IsLonValid(lon)) {
        return GetDegreeString(lon, true, compass_format, decimal_format);
    }
    return "Inv E (" + std::to_string(lon) + ")";
}

std::string GeoPos::GetLonString(double lon) { return GetLonString(lon, true, false); }
std::string GeoPos::GetLonString(bool compass_format, bool decimal_format) {
    return GetLonString(lon_, compass_format, decimal_format);
}

}  // namespace snapengine
}  // namespace alus
