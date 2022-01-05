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
#include "tie_point_grid.cuh"
#include "tie_point_grid.h"

namespace alus {
namespace snapengine {
namespace tiepointgrid {
double GetPixelDouble(double x, double y, const tiepointgrid::TiePointGrid* grid) {
    return tiepointgrid::GetPixelDoubleImpl(x, y, grid);
}
}  // namespace tiepointgrid
}  // namespace snapengine
}  // namespace alus