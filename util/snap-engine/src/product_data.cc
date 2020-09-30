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

#include <iomanip>
#include <sstream>
#include <vector>
#include <regex>

#include "product_data.h"

namespace alus::snapengine::old {

Utc::Utc(int days, int seconds, int microseconds) : days_{days}, seconds_{seconds}, microseconds_{microseconds} {}

double Utc::getMjd() const {
    return getDaysFraction() + SECONDS_TO_DAYS * (getSecondsFraction() + MICROS_TO_SECONDS * getMicroSecondsFraction());
}

std::string FormatDateString(const std::string &string) {
    std::string formatted_copy = string;
    bool first = true;
    for (auto &symbol : formatted_copy) {
        if (std::isupper(symbol) && first) {
            first = false;
        } else if (std::isupper(symbol)) {
            symbol = std::tolower(symbol);
        }
    }
    return formatted_copy;
}

Utc::Utc(const std::string &date_string) {
    std::regex pattern {R"(\d{2}-[a-zA-Z]+-\d{4} \d{2}:\d{2}:\d{2}.\d+)"};
    if (!std::regex_match(date_string, pattern)) {
        throw std::invalid_argument("Unknown date format. Date string should be in format  \"DAY-MONTH-YEAR HH:MM:SS.MICROSECONDS\"");
    }
    std::vector<std::string> tokens;
    std::stringstream ss{FormatDateString(date_string)};
    std::string token;
    while (std::getline(ss, token, '.')) {
        tokens.push_back(token);
    }
    std::stringstream date_stream{tokens[0]};
    std::tm time_struct{};
    date_stream >> std::get_time(&time_struct, TIME_FORMAT.c_str());
    time_struct.tm_year += 1900;        // get_time sets the zero year to be 1900
    time_struct.tm_mon += 1;            // corrects counting months from zero
    const double DAY_2000 = 2451544.5;  // The offset of MJD2000 from classic MJD

    int day = (1461 * (time_struct.tm_year + 4800 + (time_struct.tm_mon - 14) / 12)) / 4 +
              (367 * (time_struct.tm_mon - 2 - 12 * ((time_struct.tm_mon - 14) / 12))) / 12 -
              (3 * ((time_struct.tm_year + 4900 + (time_struct.tm_mon - 14) / 12) / 100)) / 4 + time_struct.tm_mday -
              32075 - DAY_2000;

    days_ = day;
    seconds_ = time_struct.tm_hour * 60 * 60 + time_struct.tm_min * 60 + time_struct.tm_sec;
    microseconds_ = std::stoi(tokens[1]);
}

const std::string Utc::TIME_FORMAT = "%d-%b-%Y %T";
}  // namespace alus::snapengine
