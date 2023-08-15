/**
 * ENVISAT and ERS ASAR instrument focusser for QA4EO activity (c) by CGI Estonia AS
 *
 * ENVISAT and ERS ASAR instrument focusser for QA4EO activity is licensed under a
 * Creative Commons Attribution-ShareAlike 4.0 International License.
 *
 * You should have received a copy of the license along with this
 * work. If not, see http://creativecommons.org/licenses/by-sa/4.0/
 */

#include "doris_orbit.h"

#include <cstddef>
#include <fstream>
#include <iostream>
#include <locale>
#include <numeric>
#include <sstream>
#include <string_view>

#include <boost/algorithm/string.hpp>
#include <boost/date_time/posix_time/posix_time_io.hpp>
#include <boost/date_time/posix_time/ptime.hpp>

namespace {

// Following are defined in https://earth.esa.int/eogateway/documents/20142/37627/Readme-file-for-Envisat-DORIS-POD.pdf
// chapter 2 "Doris POD data"
constexpr size_t ENVISAT_DORIS_MPH_LENGTH_BYTES{1247};
constexpr size_t ENVISAT_DORIS_SPH_LENGTH_BYTES{378};
constexpr size_t ENVISAT_DORIS_MDS_LENGTH_BYTES{204981};
constexpr size_t ENVISAT_DORIS_ORBIT_DETERMINATION_FILE_SIZE_BYTES{206606};
constexpr size_t ENVISAT_DORIS_MDSR_LENGTH_BYTES{129};
constexpr size_t ENVISAT_DORIS_MDSR_COUNT{1589};
constexpr size_t ENVISAT_DORIS_MDSR_ITEM_COUNT{11};  // Date and time are considered separate
static_assert(ENVISAT_DORIS_MDSR_COUNT * ENVISAT_DORIS_MDSR_LENGTH_BYTES == ENVISAT_DORIS_MDS_LENGTH_BYTES);
static_assert(ENVISAT_DORIS_MPH_LENGTH_BYTES + ENVISAT_DORIS_SPH_LENGTH_BYTES + ENVISAT_DORIS_MDS_LENGTH_BYTES ==
              ENVISAT_DORIS_ORBIT_DETERMINATION_FILE_SIZE_BYTES);
//constexpr std::string_view ENVISAT_DORIS_TIMESTAMP_PATTERN{"%d-%m-%Y %H:%M:%S"};
constexpr std::string_view ENVISAT_DORIS_TIMESTAMP_PATTERN{"DD-Mon-YYYY HH:MM:SS.ffffff"};
}  // namespace

namespace alus::dorisorbit {

Parsable::Parsable(std::string dsd_records) : _dsd_records{std::move(dsd_records)} {
    //    // Find the index of the last newline character
    //    size_t lastNewlinePos = _dsd_records.find_last_of('\n');
    //    lastNewlinePos = _dsd_records.find_last_of('\n', lastNewlinePos - 1);
    //
    //    const auto n_first = _dsd_records.find_first_of('\n');
    //    // Extract the first line
    //    std::string firstLine = _dsd_records.substr(0, n_first);
    //
    //    // Extract the last line
    //    std::string lastLine = _dsd_records.substr(lastNewlinePos + 1);
    //
    //    // Count the total number of rows
    //    int rowCount = std::accumulate(_dsd_records.cbegin(), _dsd_records.cend(), 0,
    //                                   [](int prev, char c) { return c != '\n' ? prev : prev + 1; });
    //
    //    // Output the results
    //    std::cout << "First Line: " << firstLine << std::endl;
    //    std::cout << "Last Line: " << lastLine << std::endl;
    //    std::cout << "Total Rows: " << rowCount << std::endl;
}

std::vector<orbitutil::OrbitInfo> Parsable::CreateOrbitInfo() const {
    std::vector<orbitutil::OrbitInfo> orbits;

    std::istringstream record_stream(_dsd_records);
    std::string record;
    // std::locale will take an ownership over this one.
    const auto* timestamp_styling = new boost::posix_time::time_input_facet(ENVISAT_DORIS_TIMESTAMP_PATTERN.data());
    const auto locale = std::locale(std::locale::classic(), timestamp_styling);
    while (std::getline(record_stream, record)) {
        std::vector<std::string> items;
        boost::split(items, record, boost::is_any_of(" \n"), boost::token_compress_on);
        if (items.size() != ENVISAT_DORIS_MDSR_ITEM_COUNT) {
            throw std::runtime_error("MDSR COUNT");
        }
        const auto item_datetime = items.front() + " " + items.at(1);//.substr(0, 8);
        std::stringstream stream(item_datetime);
        stream.imbue(locale);
        boost::posix_time::ptime date(boost::posix_time::not_a_date_time);
        stream >> date;
        std::tm timetm = boost::gregorian::to_tm(date.date());
        boost::posix_time::time_duration td = date.time_of_day();
        const auto h = td.hours();
        const auto m = td.minutes();
        const auto s = td.seconds();
        const auto ms = td.total_microseconds();
        const auto fs = td.fractional_seconds();
        (void)h;
        (void)m;
        (void)s;
        (void)ms;
        (void)fs;
        (void)timetm;
        if (date.is_not_a_date_time()) {
            throw std::runtime_error("Unparseable date '" + item_datetime +
                                     "' for format: " + std::string(ENVISAT_DORIS_TIMESTAMP_PATTERN));
        }
    }

    return orbits;
}

Parsable Parsable::TryCreateFrom(std::string_view filename) {
    std::fstream in_stream;
    in_stream.exceptions(std::ifstream::failbit | std::ifstream::badbit);
    in_stream.open(filename.data(), std::ios::in);
    in_stream.seekg(ENVISAT_DORIS_MPH_LENGTH_BYTES + ENVISAT_DORIS_SPH_LENGTH_BYTES);

    std::string dsd_records(ENVISAT_DORIS_MDS_LENGTH_BYTES, 0);
    in_stream.read(dsd_records.data(), ENVISAT_DORIS_MDS_LENGTH_BYTES);

    return Parsable(std::move(dsd_records));
}

}  // namespace alus::dorisorbit
