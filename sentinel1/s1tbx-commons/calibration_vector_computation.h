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

namespace alus {  // NOLINT TODO: concatenate namespace and remove nolint after migrating to cuda 11+
namespace s1tbx {

struct CalibrationVectorComputation {
    double time_mjd;
    const int line;
    int* pixels;
    float* sigma_nought;
    float* beta_nought;
    float* gamma;
    float* dn;
    size_t array_size;
};
}  // namespace s1tbx
}  // namespace alus