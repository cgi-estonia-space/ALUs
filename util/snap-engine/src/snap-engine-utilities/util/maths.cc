#include "maths.h"

#include <cmath>

namespace alus {
namespace snapengine {

Eigen::MatrixXd Maths::CreateVandermondeMatrix(std::vector<double> d, int warp_polynomial_order) {
    auto n = d.size();
    Eigen::MatrixXd array(n, warp_polynomial_order + 1);
    for (int i = 0; i < (int)n; i++) {
        for (int j = 0; j <= warp_polynomial_order; j++) {
            array(i, j) = pow(d.at(i), (double)j);
        }
    }
    return array;
}

std::vector<double> Maths::PolyFit(Eigen::MatrixXd A, std::vector<double> y) {
    //    todo:this might not be optimal solution, just needed it to work
    auto Q = A.householderQr();
    // std::vector to eigen vector
    double* ptr = &y[0];
    Eigen::Map<Eigen::VectorXd> yvals(ptr, y.size());
    Eigen::VectorXd result = Q.solve(yvals);
    // eigen to std::vector
    return std::vector<double>(result.data(), result.data() + result.rows() * result.cols());
}

double Maths::PolyVal(double t, std::vector<double> coeff) {
    double val = 0.0;
    int i = coeff.size() - 1;
    //        todo::looks like some logical issue because size can only be >=0
    while (i >= 0) {
        val = val * t + coeff.at(i--);
    }
    return val;
}

}  // namespace snapengine
}  // namespace alus