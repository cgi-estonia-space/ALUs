/**
* ENVISAT and ERS ASAR instrument focusser for QA4EO activity (c) by CGI Estonia AS
*
* ENVISAT and ERS ASAR instrument focusser for QA4EO activity is licensed under a
* Creative Commons Attribution-ShareAlike 4.0 International License.
*
* You should have received a copy of the license along with this
* work. If not, see http://creativecommons.org/licenses/by-sa/4.0/
*/
#pragma once

#include <string>
#include <string_view>
#include <vector>

#include "orbit_util.h"

namespace alus::dorisorbit {

class Parsable final {
public:
    static Parsable TryCreateFrom(std::string_view filename);

    std::vector<orbitutil::OrbitInfo> CreateOrbitInfo() const;

    ~Parsable() = default;

private:

    Parsable() = delete;
    Parsable(std::string dsd_records);

    std::string _dsd_records;
};

}