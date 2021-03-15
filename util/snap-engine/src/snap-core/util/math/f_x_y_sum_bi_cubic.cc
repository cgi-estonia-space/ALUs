#include "f_x_y_sum_bi_cubic.h"

namespace alus {
namespace snapengine {

BiCubic::BiCubic() : FXYSum(FXY_BI_CUBIC, 3 + 3) {}

BiCubic::BiCubic(std::vector<double> coefficients) : FXYSum(FXY_BI_CUBIC, 3 + 3, coefficients) {}

double BiCubic::ComputeZ(double x, double y) {
    std::vector<double> c = GetCoefficients();
    return c.at(0) +
           (c.at(1) +
            (c.at(3) + (c.at(6) + (c.at(10) + (c.at(13) + c.at(15) * y) * y) * y) * x +
             (c.at(7) + (c.at(11) + c.at(14) * y) * y) * y) *
                x +
            (c.at(4) + (c.at(8) + c.at(12) * y) * y) * y) *
               x +
           (c.at(2) + (c.at(5) + c.at(9) * y) * y) * y;
}

}  // namespace snapengine
}  // namespace alus
