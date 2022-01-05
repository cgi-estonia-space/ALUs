/**
 * This file is a filtered duplicate of a SNAP's
 * org.esa.snap.engine_utilities.datamodel.Orbits.java OrbitVector
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
#pragma once

#include <memory>

namespace alus::snapengine {
class OrbitVector {
public:
    double utc_mjd_;
    double x_pos_, y_pos_, z_pos_;
    double x_vel_, y_vel_, z_vel_;

    OrbitVector() = default;
    explicit OrbitVector(double utc_mjd) : utc_mjd_(utc_mjd) {}
    OrbitVector(double utc_mjd, double x_pos, double y_pos, double z_pos, double x_vel, double y_vel, double z_vel)
        : utc_mjd_(utc_mjd), x_pos_(x_pos), y_pos_(y_pos), z_pos_(z_pos), x_vel_(x_vel), y_vel_(y_vel), z_vel_(z_vel) {}

    //    todo:check if order is correct
    static int Compare(const std::shared_ptr<OrbitVector>& osv1, const std::shared_ptr<OrbitVector>& osv2);
};

}  // namespace alus::snapengine