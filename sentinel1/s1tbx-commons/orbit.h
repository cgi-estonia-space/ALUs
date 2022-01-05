/**
 * This file is a filtered duplicate of a SNAP's
 * org.jlinda.core.Orbit.java
 * ported for native code.
 * Copied from (https://github.com/senbox-org/s1tbx). It was originally stated:
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
#pragma once

#include <vector>

#include "metadata_element.h"
#include "point.h"

namespace alus::s1tbx {
class Orbit {
private:
    std::vector<double> coeff_x_, coeff_y_, coeff_z_, time_, data_x_, data_y_, data_z_;
    static const int MAXITER = 10;
    int poly_degree_;
    int num_state_vectors_;
    bool is_interpolated_ = false;
    // might need to move this to header
    static const double CRITERPOS;
    static const double CRITERTIM;

    Point RowsColumnsHeightToXyz(int rows, int columns, int height, double az_time, double rg_time,
                                 Point& ellipsoid_position);
    void ComputeCoefficients();

public:
    Orbit(const std::shared_ptr<snapengine::MetadataElement>& nest_metadata_element, int degree);
    Point RowsColumns2Xyz(int rows, int columns, double az_time, double rg_time, Point& ellipsoid_position);
    Point Xyz2T(Point point_on_ellips, double time_azimuth);
    [[nodiscard]] Point GetXyz(double az_time);
    [[nodiscard]] Point GetXyzDot(double az_time);
    [[nodiscard]] Point GetXyzDotDot(double az_time);
    static double Eq1Doppler(Point& sat_velocity, Point& point_on_ellips);
    static double Eq2Range(Point& point_ellips_sat, double rg_time);
    static double Eq3Ellipsoid(const Point& point_on_ellips, double height);
    static double Eq1DopplerDt(Point delta, Point satellite_velocity, Point satellite_acceleration);
};
}  // namespace alus::s1tbx
