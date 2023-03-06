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

#include "line_resample.h"

#include <cmath>

namespace {

[[maybe_unused]] inline void CalculateValues(int x, float* in, int in_size, float* out, int out_size) {
    (void)x;
    (void)in;
    (void)in_size;
    (void)out;
    (void)out_size;
}

}

namespace alus::lineresample {

void FillLineFrom(float* in_line, size_t in_size, float* out_line, size_t out_size) {

    float dummy;
    const auto ratio = GetRatio(in_size, out_size);
    for (size_t i = 0; i < out_size; i++) {
        const auto start_dist = (i * in_size) / static_cast<double>(out_size);
        const auto end_dist = ((i + 1) * in_size) / static_cast<double>(out_size);
        const auto in_index1 = static_cast<int>(start_dist); // Floor it.
        const auto in_index2 = static_cast<int>(end_dist);
        double index1_factor;
        double index2_factor;
        if (in_index1 == in_index2) {
            index1_factor = 1.0;
            index2_factor = 0.0;
        } else {
            float fract1 = std::modf(start_dist, &dummy);
            float fract2 = std::modf(end_dist, &dummy);
            index1_factor = (1 - fract1) / ratio;
            index2_factor = fract2 / ratio;
        }

        auto val1 = in_line[in_index1];
        auto val2 = in_line[in_index2];

        out_line[i] = val1 *  index1_factor + val2 * index2_factor;
    }
}

}