#include "utils.h"

namespace alus {

// todo: rethink where this should be
double Utils::PolyVal1D(double x, std::vector<double> coeffs) {
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

std::vector<double> Utils::Solve33(std::vector<std::vector<double>> a, std::vector<double> rhs) {
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

}  // namespace alus
