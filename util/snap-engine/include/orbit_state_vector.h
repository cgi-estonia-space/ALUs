/**
 * This file is a filtered duplicate of a SNAP's org.esa.snap.engine_utilities.datamodel.OrbitStateVector.java
 * ported for native code.
 * Copied from a snap-engine's(https://github.com/senbox-org/snap-engine) repository originally stated
 * to be implemented by "Copyright (C) 2010 Brockmann Consult GmbH (info@brockmann-consult.de)"
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

#include "product_data_utc.h"

namespace alus::snapengine {
class OrbitStateVector {
   public:
    std::shared_ptr<Utc> time_;
    double time_mjd_;
    double x_pos_, y_pos_, z_pos_;
    double x_vel_, y_vel_, z_vel_;

    OrbitStateVector() = default;
    OrbitStateVector(std::shared_ptr<Utc> t,
                     const double x_pos,
                     const double y_pos,
                     const double z_pos,
                     const double x_vel,
                     const double y_vel,
                     const double z_vel)
        : time_{t},
          time_mjd_{t->GetMjd()},
          x_pos_{x_pos},
          y_pos_{y_pos},
          z_pos_{z_pos},
          x_vel_{x_vel},
          y_vel_{y_vel},
          z_vel_{z_vel} {}
};

}  // namespace alus::snapengine
