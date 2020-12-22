#pragma once

#include <vector>

#include "eigen3/Eigen/Dense"

namespace alus {
namespace snapengine {

class Maths {
public:
    /**
     * Get Vandermonde matrix constructed from a given array.
     *
     * @param d                   The given range distance array.
     * @param warp_polynomial_order The warp polynomial order.
     * @return The Vandermonde matrix.
     */
    static Eigen::MatrixXd CreateVandermondeMatrix(std::vector<double> d, int warp_polynomial_order);

    static std::vector<double> PolyFit(Eigen::MatrixXd A, std::vector<double> y);

    static double PolyVal(double t, std::vector<double> coeff);
};

}  // namespace snapengine
}  // namespace alus