
#include "tie_point_grid.cuh"
#include "tie_point_grid.h"

namespace alus {
namespace snapengine {
namespace tiepointgrid {
double GetPixelDouble(double x, double y, const tiepointgrid::TiePointGrid *grid) {
    return tiepointgrid::GetPixelDoubleImpl(x, y, grid);
}
}  // namespace tiepointgrid
}  // namespace snapengine
}  // namespace alus