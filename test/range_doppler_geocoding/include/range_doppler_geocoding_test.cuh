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
#include "resampling.h"

namespace alus {
namespace tests {

cudaError_t LaunchGetPixelValue(dim3 grid_size,
                                dim3 block_size,
                                double azimuth_index,
                                double range_index,
                                int margin,
                                int source_image_width,
                                int source_image_height,
                                snapengine::resampling::ResamplingRaster resampling_raster,
                                snapengine::resampling::ResamplingIndex resampling_index,
                                int *sub_swath_index,
                                double *d_result);
}
}  // namespace alus