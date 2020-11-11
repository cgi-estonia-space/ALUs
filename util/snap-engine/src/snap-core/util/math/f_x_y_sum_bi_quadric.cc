#include "f_x_y_sum_bi_quadric.h"

namespace alus {
namespace snapengine {

BiQuadric::BiQuadric() : FXYSum(FXY_BI_QUADRATIC, 2 + 2) {}

BiQuadric::BiQuadric(std::vector<double> coefficients) : FXYSum(FXY_BI_QUADRATIC, 2 + 2, coefficients) {}

double BiQuadric::ComputeZ(double x, double y) {
    std::vector<double> c = GetCoefficients();
    return c.at(0) + (c.at(1) + (c.at(3) + (c.at(6) + c.at(8) * y) * y) * x + (c.at(4) + c.at(7) * y) * y) * x +
           (c.at(2) + c.at(5) * y) * y;
}

}  // namespace snapengine
}  // namespace alus
