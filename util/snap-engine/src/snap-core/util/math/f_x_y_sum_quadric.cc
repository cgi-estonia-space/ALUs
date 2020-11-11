#include "f_x_y_sum_quadric.h"

namespace alus {
namespace snapengine {

Quadric::Quadric() : FXYSum(FXY_QUADRATIC, 2) {}

Quadric::Quadric(std::vector<double> coefficients) : FXYSum(FXY_QUADRATIC, 2, coefficients) {}

double Quadric::ComputeZ(double x, double y) {
    std::vector<double> c = GetCoefficients();
    return c.at(0) + (c.at(1) + c.at(3) * x + c.at(4) * y) * x + (c.at(2) + c.at(5) * y) * y;
}

}  // namespace snapengine
}  // namespace alus
