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

namespace alus {          // NOLINT
namespace snapengine {    // NOLINT
namespace tiepointgrid {  // NOLINT

struct TiePointGrid {
    double offset_x;
    double offset_y;
    double sub_sampling_x;
    double sub_sampling_y;
    size_t grid_width;
    size_t grid_height;
    float* tie_points;
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
double GetPixelDouble(double x, double y, const tiepointgrid::TiePointGrid* grid);

}  // namespace tiepointgrid
}  // namespace snapengine
}  // namespace alus