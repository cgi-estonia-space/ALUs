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

#include <cstdlib>
#include <type_traits>

namespace alus::palsar {

void WriteRawBinary(const void* data, size_t elem_size, size_t n_elem, const char* path);

template <class T>
void WriteRawBinary(const T* data, size_t n_elem, const char* path) {
    static_assert(std::is_trivially_copyable_v<T>);
    WriteRawBinary(data, sizeof(data[0]), n_elem, path);
}
}  // namespace alus::palsar
