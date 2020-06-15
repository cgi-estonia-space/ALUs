#pragma once

#include <cuda_runtime.h>
#include "tie_point_grid.h"

namespace alus {
namespace tests {

cudaError_t LaunchGetPixelDouble(dim3 grid_size,
                                 dim3 block_size,
                                 double double_x,
                                 double double_y,
                                 double *d_result,
                                 alus::snapengine::tiepointgrid::TiePointGrid grid);
}
}  // namespace alus