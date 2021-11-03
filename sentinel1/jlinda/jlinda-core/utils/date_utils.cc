#include "jlinda/jlinda-core/utils/date_utils.h"

#include <string>
#include <string_view>
#include <vector>

#include <boost/algorithm/string/classification.hpp>
#include <boost/algorithm/string/split.hpp>

double alus::s1tbx::DateUtils::DateTimeToSecOfDay(std::string_view date_time) {
    std::vector<std::string> date_time_split_vector(2);
    boost::split(date_time_split_vector, date_time, boost::is_any_of(" "));
    auto time_hrs_min_sec = date_time_split_vector.at(1);
    std::vector<std::string> time_hrs_min_sec_vector(3);
    boost::split(time_hrs_min_sec_vector, time_hrs_min_sec, boost::is_any_of(":"));
    auto time_hrs = time_hrs_min_sec_vector.at(0);
    auto time_min = time_hrs_min_sec_vector.at(1);
    auto time_sec = time_hrs_min_sec_vector.at(2);
    return (std::stod(time_hrs) * 3600 + std::stod(time_min) * 60 + std::stod(time_sec));
}
