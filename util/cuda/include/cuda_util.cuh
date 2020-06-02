/**
 * This file is a filtered duplicate of a SNAP's SARGeocoding.java ported for native code.
 * Copied from a snap-engine's(https://github.com/senbox-org/s1tbx) repository originally stated to be implemented
 * by "Copyright (C) 2016 by Array Systems Computing Inc. http://www.array.ca"
 *
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

#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/device_vector.h>

namespace alus {
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
};
}  // namespace alus
