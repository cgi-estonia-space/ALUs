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
#include "../../../snap-engine/tie_point_grid.h"

namespace alus {
namespace tests {

cudaError_t LaunchGetPixelDouble(dim3 grid_size, dim3 block_size, double double_x, double double_y, double* d_result,
                                 alus::snapengine::tiepointgrid::TiePointGrid grid);
}
}  // namespace alus