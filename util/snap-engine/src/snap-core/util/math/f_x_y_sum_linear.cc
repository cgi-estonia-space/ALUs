#include "f_x_y_sum_linear.h"

namespace alus {
namespace snapengine {

Linear::Linear() : FXYSum(FXY_LINEAR, 1) {}

Linear::Linear(std::vector<double> coefficients) : FXYSum(FXY_LINEAR, 1, coefficients) {}

double Linear::ComputeZ(double x, double y) {
    std::vector<double> c = GetCoefficients();
    return c.at(0) + c.at(1) * x + c.at(2) * y;
}

}  // namespace snapengine
}  // namespace alus
