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

#include <cuda_runtime.h>

namespace alus {
namespace tests {

struct EGM96data {
    int max_lats;
    int max_lons;
    float* egm;
    int size;
};

cudaError_t LaunchEGM96(dim3 grid_size, dim3 block_size, double* lats, double* lons, float* results, EGM96data data);

}  // namespace tests
}  // namespace alus
