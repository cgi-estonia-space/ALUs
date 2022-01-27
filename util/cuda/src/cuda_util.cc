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

#include "cuda_util.h"

namespace alus {  // NOLINT TODO: concatenate namespace and remove nolint after migrating to cuda 11+
namespace cuda {

// DO NOT USE math::ceil here. it was removed because of its inaccuracy.
int GetGridDim(int blockDim, int dataDim) {
    double temp = dataDim / blockDim;  // NOLINT
    int temp_int;
    if (temp < 1) {
        return 1;
    }
    temp_int = static_cast<int>(temp);
    if (temp_int * blockDim < dataDim) {
        temp_int++;
    }
    return temp_int;
}

}  // namespace cuda
}  // namespace alus
