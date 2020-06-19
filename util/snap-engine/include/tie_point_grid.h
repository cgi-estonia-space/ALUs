#pragma once

#include "cuda_util.cuh"

namespace alus {
namespace snapengine {
namespace tiepointgrid {

struct TiePointGrid {
    double offset_x;
    double offset_y;
    double sub_sampling_x;
    double sub_sampling_y;
    int grid_width;
    int grid_height;
    cudautil::KernelArray<float> tie_points;
};

/**
 * Gets the interpolated sample for the pixel located at (x,y) as a double value. <p>
 * <p>
 * If the pixel co-ordinates given by (x,y) are not covered by this tie-point grid, the method extrapolates.
 *
 * @param x The X co-ordinate of the pixel location, given in the pixel co-ordinates of the data product to which
 *          this tie-point grid belongs to.
 * @param y The Y co-ordinate of the pixel location, given in the pixel co-ordinates of the data product to which
 *          this tie-point grid belongs to.
 * @todo there is a check for discontinuity in SNAP code, yet any discontinuity other than 0 was never encountered
 *          (this check should be implemented later and discontinuity should be added either to function parameters
 *          or to TiePointGrid struct)
 */
double GetPixelDouble(double x, double y, const tiepointgrid::TiePointGrid *grid);

}  // namespace tiepointgrid
}  // namespace snapengine
}  // namespace alus