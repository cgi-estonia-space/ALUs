/**
 * ENVISAT and ERS ASAR instrument focusser for QA4EO activity (c) by CGI Estonia AS
 *
 * ENVISAT and ERS ASAR instrument focusser for QA4EO activity is licensed under a
 * Creative Commons Attribution-ShareAlike 4.0 International License.
 *
 * You should have received a copy of the license along with this
 * work. If not, see http://creativecommons.org/licenses/by-sa/4.0/
 */

#include <iostream>
#include <stdexcept>
#include <string_view>

#include "doris_orbit.h"
#include "orbit_util.h"

int main(int argc, char* argv[]) {
    (void)argc;
    (void)argv;

    constexpr std::string_view filename{
        "/home/sven/temp/envisat_ers_orbit/envisat/DOR_VOR_AX_d/200204/"
        "DOR_VOR_AXVF-P20120425_064700_20020428_215528_20020430_002328"};

    try {
        const auto orb = alus::dorisorbit::Parsable::TryCreateFrom(filename);
        const auto orbits = orb.CreateOrbitInfo();
    } catch (const std::runtime_error& e) {
        std::cerr << "Caught error - " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "Caught unknown error" << std::endl;
        return 2;
    }

    return 0;
}