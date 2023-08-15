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

namespace alus::orbitutil {

struct OrbitInfo {
    double time_point;
    double x_pos;
    double y_pos;
    double z_pos;
    double x_vel;
    double y_vel;
    double z_vel;

    void Print() const
    {
        printf("OSV=%f,%f,%f,%f,%f,%f,%f\n", time_point, x_pos, y_pos, z_pos, x_vel, y_vel, z_vel);
    }
};

}