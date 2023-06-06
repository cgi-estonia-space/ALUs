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

#include <cmath>

#include "cuda_stubs.h"

namespace alus::math::calcfuncs{

inline DEVICE_STUB HOST_STUB float Decibel(float value, float no_data_value = std::nanf("")) {
    if (value == 0 || isnan(value)) {
        return value;
    }
    if (!isnan(no_data_value) && value == no_data_value) {
        return value;
    }

    return 10 * log10(value);
}

}