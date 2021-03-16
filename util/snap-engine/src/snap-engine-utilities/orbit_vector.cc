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
#include "snap-engine-utilities/orbit_vector.h"

namespace alus {
namespace snapengine {

int OrbitVector::Compare(const std::shared_ptr<OrbitVector>& osv1, const std::shared_ptr<OrbitVector>& osv2) {
    //    if (osv1->utc_mjd_ < osv2->utc_mjd_) {
    //        return -1;
    //    } else if (osv1->utc_mjd_ > osv2->utc_mjd_) {
    //        return 1;
    //    } else {
    //        return 0;
    //    }
    return static_cast<int>(osv1->utc_mjd_ < osv2->utc_mjd_ && osv2->utc_mjd_ >= osv1->utc_mjd_);
}

}  // namespace snapengine
}  // namespace alus
