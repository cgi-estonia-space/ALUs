/**
 * This file is a filtered duplicate of a SNAP's
 * org.esa.snap.core.util.math.Approximator.java
 * ported for native code.
 * Copied from (https://github.com/senbox-org/snap-engine). It was originally stated:
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the Free
 * Software Foundation; either version 3 of the License, or (at your option)
 * any later version.
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for
 * more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, see http://www.gnu.org/licenses/
 */
#include "snap-core/core/util/math/approximator.h"

#include <cmath>
#include <stdexcept>

namespace alus::snapengine {

// void Approximator::ApproximateFXY(std::vector<std::vector<double>> data, std::vector<int> indices,
//                                  std::vector<std::shared_ptr<IFXY>> f, std::vector<double> c) {
//    auto n = f.size();
//    std::vector<std::vector<double>> a(n, std::vector<double>(n));
//    std::vector<double> b(n);
//    double x;
//    double y;
//    double z;
//    int i_x = 0;
//    int i_y = 1;
//    int i_z = 2;
//    if (!indices.empty()) {
//        i_x = indices.at(0);
//        i_y = indices.at(1);
//        i_z = indices.at(2);
//    }
//    for (std::size_t i = 0; i < n; i++) {      // Rows i=1..n
//        for (std::size_t j = i; j < n; j++) {  // Columns j=1..n
//            for (auto point : data) {
//                x = point.at(i_x);
//                y = point.at(i_y);
//                double result = f.at(i)->F(x, y) * f.at(j)->F(x, y);
//                if (!std::isnan(result)) {
//                    a.at(i).at(j) += result;  // sum fi(x,y) * fj(x,y)
//                }
//            }
//        }
//        // Copy, since matrix is symetric
//        for (std::size_t j = 0; j < i; j++) {  // Columns j=1..i-1
//            a.at(i).at(j) = a.at(j).at(i);
//        }
//        // Column n+1
//        for (auto point : data) {
//            x = point.at(i_x);
//            y = point.at(i_y);
//            z = point.at(i_z);
//            double result = z * f.at(i)->F(x, y);
//            if (!std::isnan(result)) {
//                b.at(i) += result;  // sum z * fi(x,y)
//            }
//        }
//    }
//    Solve2(a, b, c);
//}
// try to use eigen directly, std::vector is not efficient for 2d matrix
void Approximator::ApproximateFXY(const std::vector<std::vector<double>>& data, std::vector<int> indices,
                                  const std::vector<std::reference_wrapper<IFXY>>& f, std::vector<double>& c) {
    auto n = f.size();
    Eigen::MatrixXd a = Eigen::MatrixXd::Zero(n, n);
    Eigen::VectorXd b = Eigen::VectorXd::Zero(n);
    double x;
    double y;
    double z;
    int i_x = 0;
    int i_y = 1;
    int i_z = 2;
    if (!indices.empty()) {
        i_x = indices.at(0);
        i_y = indices.at(1);
        i_z = indices.at(2);
    }
    for (std::size_t i = 0; i < n; i++) {      // Rows i=1..n
        for (std::size_t j = i; j < n; j++) {  // Columns j=1..n
            for (auto point : data) {
                x = point.at(i_x);
                y = point.at(i_y);
                double result = f.at(i).get().F(x, y) * f.at(j).get().F(x, y);
                if (!std::isnan(result)) {
                    a(i, j) += result;  // sum fi(x,y) * fj(x,y)
                }
            }
        }
        // Copy, since matrix is symetric
        for (std::size_t j = 0; j < i; j++) {  // Columns j=1..i-1
            a(i, j) = a(j, i);
        }
        // Column n+1
        for (const auto& point : data) {
            x = point.at(i_x);
            y = point.at(i_y);
            z = point.at(i_z);
            double result = z * f.at(i).get().F(x, y);
            if (!std::isnan(result)) {
                b(i) += result;  // sum z * fi(x,y)
            }
        }
    }
    Eigen::VectorXd c2 = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(c.data(), c.size());
    Solve2(a, b, c2);
}

void Approximator::Solve2(const Eigen::MatrixXd& a, const Eigen::VectorXd& b, Eigen::VectorXd& x) {
    if (a.determinant() == 0.0 || std::isnan(a.determinant()) || std::isinf(a.determinant())) {
        throw std::runtime_error("Expected an invertible matrix, but matrix is degenerate: det = " +
                                 std::to_string(a.determinant()));
    }
    x = a.bdcSvd().solve(b);
}

std::vector<double> Approximator::ComputeErrorStatistics(const std::vector<std::vector<double>>& data,
                                                         std::vector<int> indices,
                                                         const std::vector<std::reference_wrapper<IFXY>>& f,
                                                         const std::vector<double>& c) {
    auto m = data.size();
    double x;
    double y;
    double z;
    double d;
    double mse = 0.0;
    double emax = 0.0;
    int i_x = 0;
    int i_y = 1;
    int i_z = 2;
    if (!indices.empty()) {
        i_x = indices.at(0);
        i_y = indices.at(1);
        i_z = indices.at(2);
    }
    for (auto point : data) {
        x = point.at(i_x);
        y = point.at(i_y);
        z = point.at(i_z);
        d = FXYSum::ComputeZ(f, c, x, y) - z;
        emax = std::max(emax, std::abs(d));
        mse += d * d;
    }
    mse /= m;
    double rmse = sqrt(mse);
    return std::vector<double>{rmse, emax};
}

// void Approximator::Solve2(std::vector<std::vector<double>> a, std::vector<double> b, std::vector<double> x) {
//    //provide solution using eigen library
//    //try to check results
//    double *v = &param[n];
//    Eigen::Map<Eigen::MatrixXd> matrix(v,n + n * n,1);
//
//
//    auto m = b.size();
//    auto n = x.size();
//
//    final SingularValueDecomposition svd;
//    final Matrix u;
//    final Matrix v;
//
//    //assumption which we need to check over:
//    //m rows
//    //n columns
//    //a = matrix with m rows n columns
//
//    Eigen::MatrixXd A(a, m, n);
//    final Matrix matrix = new Matrix(a, m, n);
//    final double det = matrix.det();
//    if (det == 0.0 || std::isnan(det) || std::isinf(det)) {
//        throw std::runtime_error("Expected an invertible matrix, but matrix is degenerate: det = " + det);
//    }
//
//    svd = matrix.svd();
//    u = svd.getU();
//    v = svd.getV();
//
//    final double[] s = svd.getSingularValues();
//    final int rank = svd.rank();
//
//    for (int j = 0; j < rank; j++) {
//        x[j] = 0.0;
//        for (int i = 0; i < m; i++) {
//            x[j] += u.get(i, j) * b[i];
//        }
//        s[j] = x[j] / s[j];
//    }
//    for (int j = 0; j < n; j++) {
//        x[j] = 0.0;
//        for (int i = 0; i < rank; i++) {
//            x[j] += v.get(j, i) * s[i];
//        }
//    }
//}

}  // namespace alus::snapengine