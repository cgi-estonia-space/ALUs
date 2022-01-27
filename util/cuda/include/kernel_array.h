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

namespace alus {  // NOLINT TODO: clang-tidy warns about possible exceptions. This should be addressed outside of the
                  // current task.
namespace cuda {
/**
 * Thrust containers' wrapper.
 *
 * CUDA kernels do not accept Thrust containers as arguments. This solves the problem.
 * Solution inspired by https://codeyarns.com/2011/04/09/how-to-pass-thrust-device-vector-to-kernel/
 */
template <typename T>
struct KernelArray {
    T* array;
    size_t size;

    [[nodiscard]] size_t ByteSize() const { return sizeof(T) * size; }
};

}  // namespace cuda
}  // namespace alus