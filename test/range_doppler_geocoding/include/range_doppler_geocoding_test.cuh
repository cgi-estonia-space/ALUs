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
                                alus::snapengine::resampling::TileData *tile_data,
                                double *band_data_buffer,
                                int *sub_swath_index,
                                double *d_result);
}
}  // namespace alus