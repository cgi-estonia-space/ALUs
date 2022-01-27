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
#include "jlinda/jlinda-core/utils/date_utils.h"

#include <string>
#include <string_view>
#include <vector>

#include <boost/algorithm/string/classification.hpp>
#include <boost/algorithm/string/split.hpp>

double alus::s1tbx::DateUtils::DateTimeToSecOfDay(std::string_view date_time) {
    const int seconds_in_minute{60};
    const int seconds_in_hour{3600};

    std::vector<std::string> date_time_split_vector(2);
    boost::split(date_time_split_vector, date_time, boost::is_any_of(" "));
    auto time_hrs_min_sec = date_time_split_vector.at(1);
    std::vector<std::string> time_hrs_min_sec_vector(3);  // NOLINT
    boost::split(time_hrs_min_sec_vector, time_hrs_min_sec, boost::is_any_of(":"));
    auto time_hrs = time_hrs_min_sec_vector.at(0);
    auto time_min = time_hrs_min_sec_vector.at(1);
    auto time_sec = time_hrs_min_sec_vector.at(2);
    return (std::stod(time_hrs) * seconds_in_hour + std::stod(time_min) * seconds_in_minute + std::stod(time_sec));
}
