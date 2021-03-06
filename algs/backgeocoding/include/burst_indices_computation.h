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

namespace alus {
namespace backgeocoding {

struct BurstIndices {
    int first_burst_index;
    int second_burst_index;
    bool in_upper_part_of_first_burst;
    bool in_upper_part_of_second_burst;
    bool valid;
};
}  // namespace backgeocoding
}  // namespace alus