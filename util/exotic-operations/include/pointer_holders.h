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

namespace alus {

/**
    The whole point of this is to store a matrix or a cube of data on the gpu.
    The problem is that in some cases, we can have several of these data piles and they can have
    different sizes and meanings. This is also a reason why there is an ID field here, as it may
    be necessary to tell different data piles apart, that are entered in a random order.

    Use the x, y and z to describe the dimensions of your matrix that will be under the pointer variable.
*/
struct PointerHolder {
    int id;
    void* pointer;
    int x;
    int y;
    int z;
};

struct PointerArray {
    const PointerHolder* array;
    size_t size;
};
}  // namespace alus
