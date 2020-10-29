#include "poly_utils.h"

namespace alus {

// todo: rethink where this should be
double PolyUtils::PolyVal1D(double x, std::vector<double> coeffs) {
    double sum = 0.0;
    for (std::vector<double>::reverse_iterator it = coeffs.rbegin(); it != coeffs.rend(); it++) {
        sum *= x;
        sum += *it;
        //    coeffs.at(d);
    }
    //  for (size_t d = (coeffs.size() - 1); d >= 0; --d) {
    //  for (size_t d = (coeffs.size() - 1); d >= 0; --d) {
    //    sum *= x;
    //    sum += coeffs.at(d);
    //  }
    return sum;
}

std::vector<double> PolyUtils::Solve33(std::vector<std::vector<double>> a, std::vector<double> rhs) {
    std::vector<double> result(3);

    if (a[0].size() != 3 || a.size() != 3) {
        throw std::invalid_argument("Solve33: input: size of a not 33.");
    }
    if (rhs.size() != 3) {
        throw std::invalid_argument("Solve33: input: size rhs not 3x1.");
    }

    // real8 l_10, l_20, l_21: used lower matrix elements
    // real8 u_11, u_12, u_22: used upper matrix elements
    // real8 b_0,  b_1,  b_2:  used Ux=b
    double l_10 = a[1][0] / a[0][0];
    double l_20 = a[2][0] / a[0][0];
    double u_11 = a[1][1] - l_10 * a[0][1];
    double l_21 = (a[2][1] - (a[0][1] * l_20)) / u_11;
    double u_12 = a[1][2] - l_10 * a[0][2];
    double u_22 = a[2][2] - l_20 * a[0][2] - l_21 * u_12;

    // ______ Solution: forward substitution ______
    double b_0 = rhs[0];
    double b_1 = rhs[1] - b_0 * l_10;
    double b_2 = rhs[2] - b_0 * l_20 - b_1 * l_21;

    // ______ Solution: backwards substitution ______
    result[2] = b_2 / u_22;
    result[1] = (b_1 - u_12 * result[2]) / u_11;
    result[0] = (b_0 - a[0][1] * result[1] - a[0][2] * result[2]) / a[0][0];

    return result;
}

Eigen::VectorXd PolyUtils::PolyFit(Eigen::VectorXd xvals, Eigen::VectorXd yvals, int order) {
    // todo: assert(xvals.size() == yvals.size());
    // todo: assert(order >= 1 && order <= xvals.size() - 1);
    Eigen::MatrixXd A(xvals.size(), order + 1);

    for (int i = 0; i < xvals.size(); i++) {
        A(i, 0) = 1.0;
    }

    for (int j = 0; j < xvals.size(); j++) {
        for (int i = 0; i < order; i++) {
            A(j, i + 1) = A(j, i) * xvals(j);
        }
    }

    auto Q = A.householderQr();
    auto result = Q.solve(yvals);
    return result;
}

Eigen::VectorXd PolyUtils::Normalize(Eigen::VectorXd t) {
    int i = t.size() / 2;
    return (t - (t(i) * Eigen::VectorXd::Ones(t.size()))) / 10.0;
}

Eigen::VectorXd PolyUtils::PolyFitNormalized(Eigen::VectorXd t, Eigen::VectorXd y, int degree) {
    return PolyFit(Normalize(t), y, degree);
}

std::vector<double> PolyUtils::PolyFitNormalized(std::vector<double> t, std::vector<double> y, int degree) {
    auto result = PolyFit(Normalize(Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(t.data(), t.size())),
                          Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(y.data(), y.size()),
                          degree);
    return std::vector<double>(result.data(), result.data() + result.rows() * result.cols());
}

}  // namespace alus
