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
#include <type_traits>

namespace alus {

struct TypeParameters {
    template <typename T>
    static TypeParameters CreateFor() {
        return {sizeof(T), std::is_signed_v<T>, std::is_floating_point_v<T>};
    }

    size_t size_bytes;
    bool is_signed;
    bool is_float;
};

}  // namespace alus
