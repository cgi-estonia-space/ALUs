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
#pragma once

#include <cstddef>
#include <vector>

namespace alus::s1tbx {
/**
 * Port of SNAP's CalibrationVector.
 *
 * Original implementation can be found in Sentinel1Utils.java
 */
struct CalibrationVector {
    const double time_mjd;
    const int line;
    const std::vector<int> pixels;
    const std::vector<float> sigma_nought;
    const std::vector<float> beta_nought;
    const std::vector<float> gamma;
    const std::vector<float> dn;
    const size_t array_size;
};
}  // namespace alus::s1tbx
