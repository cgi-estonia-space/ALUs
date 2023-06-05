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

#include "kernel_array.h"

namespace alus {

struct TcTileIndexPair {
    int source_x_0;
    int source_y_0;
    size_t source_width;
    size_t source_height;
    int target_x_0;
    int target_y_0;
    size_t target_width;
    size_t target_height;
};

}  // namespace alus