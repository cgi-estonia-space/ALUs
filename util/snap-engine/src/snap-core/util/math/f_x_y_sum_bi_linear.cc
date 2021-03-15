#include "f_x_y_sum_bi_linear.h"

namespace alus {
namespace snapengine {

BiLinear::BiLinear() : FXYSum(FXY_BI_LINEAR, 1 + 1) {}

BiLinear::BiLinear(std::vector<double> coefficients) : FXYSum(FXY_BI_LINEAR, 1 + 1, coefficients) {}

double BiLinear::ComputeZ(double x, double y) {
    std::vector<double> c = GetCoefficients();
    return c.at(0) + (c.at(1) + c.at(3) * y) * x + c.at(2) * y;
}

}  // namespace snapengine
}  // namespace alus