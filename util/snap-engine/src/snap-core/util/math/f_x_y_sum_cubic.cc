#include "f_x_y_sum_cubic.h"

namespace alus {
namespace snapengine {

Cubic::Cubic() : FXYSum(FXY_CUBIC, 3) {}

Cubic::Cubic(std::vector<double> coefficients) : FXYSum(FXY_CUBIC, 3, coefficients) {}

double Cubic::ComputeZ(double x, double y) {
    std::vector<double> c = GetCoefficients();
    return c.at(0) + (c.at(1) + (c.at(3) + c.at(6) * x + c.at(7) * y) * x + (c.at(4) + c.at(8) * y) * y) * x +
           (c.at(2) + (c.at(5) + c.at(9) * y) * y) * y;
}

}  // namespace snapengine
}  // namespace alus
