#include "tie_point_grid.cuh"
#include "tie_point_grid_test.cuh"

namespace alus {
namespace tests {

__global__ void TiePointGridTester(double double_x,
                                   double double_y,
                                   double *d_result,
                                   alus::snapengine::tiepointgrid::TiePointGrid grid) {
    *d_result = snapengine::tiepointgrid::GetPixelDoubleImpl(double_x, double_y, &grid);
}

cudaError_t LaunchGetPixelDouble(dim3 grid_size,
                                 dim3 block_size,
                                 double double_x,
                                 double double_y,
                                 double *d_result,
                                 alus::snapengine::tiepointgrid::TiePointGrid grid) {
    TiePointGridTester<<<grid_size, block_size>>>(double_x, double_y, d_result, grid);

    return cudaGetLastError();
}

}  // namespace tests
}  // namespace alus