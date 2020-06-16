#pragma once

#include <stdexcept>
#include <vector>

namespace alus {

class Utils {
   public:
    [[nodiscard]] static double PolyVal1D(double x, std::vector<double> coeffs);
    [[nodiscard]] static std::vector<double> Solve33(std::vector<std::vector<double>> a, std::vector<double> rhs);
};

}  // namespace alus
