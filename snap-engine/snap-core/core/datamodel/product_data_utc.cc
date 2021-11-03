/**
 * This file is a filtered duplicate of a SNAP's
 * static nested class Utc which is inside org.esa.snap.core.datamodel.ProductData.java
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
#include "product_data_utc.h"

#include <cmath>
#include <locale>
#include <sstream>
#include <stdexcept>

#include "../util/guardian.h"
#include "parse_exception.h"

namespace alus {
namespace snapengine {

Utc::Utc() : UInt(3) {}

Utc::Utc(std::vector<uint32_t> elems) : Utc(elems.at(0), elems.at(1), elems.at(2)) {}

Utc::Utc(int days, int seconds, int microseconds) : UInt(3) {
    SetElemIntAt(0, days);
    SetElemIntAt(1, seconds);
    SetElemIntAt(2, microseconds);
}

Utc::Utc(double mjd) : UInt(3) {
    double micro_seconds = std::fmod((mjd * SECONDS_PER_DAY * MICROS_PER_SECOND), MICROS_PER_SECOND);
    double seconds = std::fmod((mjd * SECONDS_PER_DAY), SECONDS_PER_DAY);
    const double days = static_cast<int>(mjd);

    if (micro_seconds < 0) {  // handle date prior to year 2000
        micro_seconds += MICROS_PER_SECOND;
        seconds -= 1;
    }

    SetElemUIntAt(0, days);
    SetElemUIntAt(1, seconds);
    SetElemUIntAt(2, micro_seconds);
}

double Utc::GetMjd() const {
    return GetDaysFraction() + SECONDS_TO_DAYS * (GetSecondsFraction() + MICROS_TO_SECONDS * GetMicroSecondsFraction());
}
std::shared_ptr<Utc> Utc::Create(boost::posix_time::ptime date_time, long micros) {
    // 1)get inital date from which to measure milliseconds from (2000) (java has CreateCalendar)
    boost::posix_time::ptime start_date(CreateCalendar());
    // 2)get difference between two dates(in millis) in days, seconds and micros
    auto diff = (date_time - start_date).total_milliseconds();

    int millis_per_second = 1000;
    int millis_per_day = 24 * 60 * 60 * millis_per_second;
    long days = diff / millis_per_day;
    long seconds = (diff - days * millis_per_day) / millis_per_second;

    return std::make_shared<Utc>(static_cast<int>(days), static_cast<int>(seconds), static_cast<int>(micros));
}
// boost::posix_time::ptime Utc::CreateCalendar() {
//    // todo: might need to add some custom locale (check java impl)
//
//    return boost::posix_time::ptime(boost::gregorian::date{2000, 1, 1});
//}
boost::posix_time::ptime Utc::GetAsDate() { return GetAsCalendar(); }

boost::gregorian::date Utc::CreateCalendar() {
    // todo: might need to add some custom locale (check java impl)

    return boost::gregorian::date{2000, 1, 1};
}

boost::posix_time::ptime Utc::GetAsCalendar() const {
    auto start_date_time = boost::posix_time::ptime(CreateCalendar());
    auto seconds = boost::posix_time::seconds(GetSecondsFraction());
    auto millis = boost::posix_time::milliseconds(static_cast<int>(std::round(GetMicroSecondsFraction() / 1000.0)));
    auto days = boost::gregorian::days(GetDaysFraction());

    //    todo: return format! boost::gregorian::date{2000, 1, 1}
    return start_date_time + days + seconds + millis;
}

std::string Utc::Format() const {
    return Format(DATE_FORMAT_PATTERN);
}

std::string Utc::Format(std::string_view format) const {
    auto start_date_time = boost::posix_time::ptime(CreateCalendar());
    auto days = boost::gregorian::days(GetDaysFraction());
    auto seconds = boost::posix_time::seconds(GetSecondsFraction());
    // ptime to stringstream to string.
    std::stringstream stream;
    stream.imbue(std::locale(std::locale::classic(), CreateDateFormatOut(format)));
    stream << (start_date_time + days + seconds) << ".";
    std::string micros_string = std::to_string(GetMicroSecondsFraction());
    for (int i = micros_string.length(); i < 6; i++) {
        stream << ('0');
    }
    stream << micros_string;
    std::string out = stream.str();
    for (auto& c : out) c = toupper(c);
    return out;
}
boost::posix_time::time_facet* Utc::CreateDateFormatOut(std::string_view pattern) {
    return new boost::posix_time::time_facet(std::string(pattern).data());
}

boost::posix_time::time_input_facet* Utc::CreateDateFormatIn(std::string_view pattern) {
    return new boost::posix_time::time_input_facet(std::string(pattern).data());
}
std::shared_ptr<Utc> Utc::Parse(std::string_view text) { return Parse(text, DATE_FORMAT_PATTERN); }


std::shared_ptr<Utc> Utc::Parse(const std::string_view text, std::string_view pattern) {
    Guardian::AssertNotNullOrEmpty("text", text);
    //Guardian::AssertNotNull("date_time_format", date_time_format);

    auto dot_pos = text.find_last_of(".");
    std::string_view no_fraction_string = text;
    int64_t micros = 0;

    if (dot_pos != std::string::npos) {
        no_fraction_string = text.substr(0, dot_pos);
        std::string_view fraction_string = text.substr(dot_pos + 1, text.length());
        if (fraction_string.length() > 6) {  // max. 6 digits!
            throw alus::ParseException("Unparseable date:" + std::string{text} +
                                       " at position: " + std::to_string(dot_pos));
        }
        try {
            micros = std::stoi(std::string(fraction_string));
        } catch (const std::exception& e) {
            throw alus::ParseException("Unparseable date:" + std::string{text} +
                                       " at position: " + std::to_string(dot_pos) + ", reason: " + e.what());
        }

        for (auto i = fraction_string.length(); i < 6; i++) {
            micros *= 10;
        }
    }
    std::stringstream stream(std::string{no_fraction_string});
    stream.imbue(std::locale(std::locale::classic(), CreateDateFormatIn(pattern)));
    boost::posix_time::ptime date(boost::posix_time::not_a_date_time);
    stream >> date;
    if(date.is_not_a_date_time())
    {
        throw alus::ParseException("Unparseable date:" + std::string(text) + " for format: " + std::string(pattern));
    }

    return Create(date, micros);
}
std::string Utc::GetElemString() { return Format(); }
std::shared_ptr<ProductData> Utc::CreateDeepClone() const {
    std::shared_ptr<Utc> data = std::make_shared<Utc>(array_.size());
    std::copy(array_.begin(), array_.end(), data->array_.begin());
    return data;
}

}  // namespace snapengine
}  // namespace alus
