/**
 * This file is a filtered duplicate of a SNAP's
 * org.esa.snap.core.util.math.FXYSum.java
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
#include "snap-core/util/math/f_x_y_sum.h"

#include "snap-core/util/math/approximator.h"
#include "snap-core/util/math/f_x_y_sum_bi_cubic.h"
#include "snap-core/util/math/f_x_y_sum_bi_linear.h"
#include "snap-core/util/math/f_x_y_sum_bi_quadric.h"
#include "snap-core/util/math/f_x_y_sum_cubic.h"
#include "snap-core/util/math/f_x_y_sum_linear.h"
#include "snap-core/util/math/f_x_y_sum_quadric.h"

namespace alus {
namespace snapengine {

FXYSum::FXYSum(const std::vector<std::reference_wrapper<IFXY>>& functions) : FXYSum(functions, -1) {}

FXYSum::FXYSum(const std::vector<std::reference_wrapper<IFXY>>& functions, int order)
    : FXYSum(functions, order, std::vector<double>(0)) {}

FXYSum::FXYSum(const std::vector<std::reference_wrapper<IFXY>>& functions, int order,
               const std::vector<double>& coefficients) {
    if (functions.empty()) {
        throw std::invalid_argument("'functions' is null or empty");
    }
    _f_ = functions;
    if (coefficients.empty()) {
        _c_ = std::vector<double>(functions.size());
    } else {
        if (functions.size() != coefficients.size()) {
            throw std::invalid_argument("'functions.length' != 'coefficients.length'");
        }
        _c_ = coefficients;
    }
    _order_ = order;
}

std::shared_ptr<FXYSum> FXYSum::CreateFXYSum(int order, const std::vector<double>& coefficients) {
    std::shared_ptr<FXYSum> sum;
    switch (order) {
        case 1:
            if (coefficients.size() == 3) {
                sum = std::make_shared<Linear>(coefficients);
            } else {
                sum = nullptr;
            }
            break;
        case 2:
            if (coefficients.size() == 4) {
                sum = std::make_shared<BiLinear>(coefficients);
            } else if (coefficients.size() == 6) {
                sum = std::make_shared<Quadric>(coefficients);
            } else {
                sum = nullptr;
            }
            break;
        case 3:
            if (coefficients.size() == 10) {
                sum = std::make_shared<Cubic>(coefficients);
            } else {
                sum = nullptr;
            }
            break;
        case 4:
            if (coefficients.size() == 9) {
                sum = std::make_shared<BiQuadric>(coefficients);
            } else {
                sum = nullptr;
            }
            break;
        case 6:
            if (coefficients.size() == 16) {
                sum = std::make_shared<BiCubic>(coefficients);
            } else {
                sum = nullptr;
            }
            break;
        default:
            sum = nullptr;
            break;
    }
    return sum;
}

std::shared_ptr<FXYSum> FXYSum::CreateCopy(const std::shared_ptr<FXYSum>& fxy_sum) {
    std::vector<double> coefficients(fxy_sum->GetCoefficients().size());
    std::copy(fxy_sum->GetCoefficients().begin(), fxy_sum->GetCoefficients().end(), coefficients.begin());
    auto fxy_sum_copy = std::make_shared<FXYSum>(fxy_sum->GetFunctions(), fxy_sum->GetOrder(), coefficients);
    return fxy_sum_copy;
}

double FXYSum::ComputeZ(const std::vector<std::reference_wrapper<IFXY>>& f, const std::vector<double>& c, double x,
                        double y) {
    auto n = f.size();
    double z = 0.0;
    for (std::size_t i = 0; i < n; i++) {
        z += c.at(i) * f.at(i).get().F(x, y);
    }
    return z;
}

void FXYSum::Approximate(const std::vector<std::vector<double>>& data, const std::vector<int>& indices) {
    //    todo: check if this works like snap solution
    Approximator::ApproximateFXY(data, indices, _f_, _c_);
    _error_statistics_ = Approximator::ComputeErrorStatistics(data, indices, _f_, _c_);
}

std::vector<std::reference_wrapper<IFXY>> FXYSum::FXY_LINEAR{/*0*/ IFXY::ONE, /*1*/ IFXY::X, IFXY::Y};
std::vector<std::reference_wrapper<IFXY>> FXYSum::FXY_BI_LINEAR{/*0*/ IFXY::ONE, /*1*/ IFXY::X, IFXY::Y,
                                                                /*2*/ IFXY::XY};
std::vector<std::reference_wrapper<IFXY>> FXYSum::FXY_QUADRATIC{/*0*/ IFXY::ONE, /*1*/ IFXY::X, IFXY::Y,
                                                                /*2*/ IFXY::X2,  IFXY::XY,      IFXY::Y2};
std::vector<std::reference_wrapper<IFXY>> FXYSum::FXY_BI_QUADRATIC{
    /*0*/ IFXY::ONE, /*1*/ IFXY::X, IFXY::Y,
    /*2*/ IFXY::X2,  IFXY::XY,      IFXY::Y2,
    /*3*/ IFXY::X2Y, IFXY::XY2,     /*4*/ IFXY::X2Y2};
std::vector<std::reference_wrapper<IFXY>> FXYSum::FXY_CUBIC{
    /*0*/ IFXY::ONE, /*1*/ IFXY::X,  IFXY::Y,   /*2*/ IFXY::X2, IFXY::XY,
    IFXY::Y2,        /*3*/ IFXY::X3, IFXY::X2Y, IFXY::XY2,      IFXY::Y3};

std::vector<std::reference_wrapper<IFXY>> FXYSum::FXY_BI_CUBIC{
    /*0*/ IFXY::ONE, /*1*/ IFXY::X,    IFXY::Y,    /*2*/ IFXY::X2,  IFXY::XY,        IFXY::Y2,
    /*3*/ IFXY::X3,  IFXY::X2Y,        IFXY::XY2,  IFXY::Y3,        /*4*/ IFXY::X3Y, IFXY::X2Y2,
    IFXY::XY3,       /*5*/ IFXY::X3Y2, IFXY::X2Y3, /*6*/ IFXY::X3Y3};
std::vector<std::reference_wrapper<IFXY>> FXYSum::FXY_4TH{
    /*0*/ IFXY::ONE, /*1*/ IFXY::X,  IFXY::Y,    /*2*/ IFXY::X2, IFXY::XY,
    IFXY::Y2,        /*3*/ IFXY::X3, IFXY::X2Y,  IFXY::XY2,      IFXY::Y3,
    /*4*/ IFXY::X4,  IFXY::X3Y,      IFXY::X2Y2, IFXY::XY3,      IFXY::Y4};
std::vector<std::reference_wrapper<IFXY>> FXYSum::FXY_BI_4TH{
    /*0*/ IFXY::ONE, /*1*/ IFXY::X,    IFXY::Y,    /*2*/ IFXY::X2,  IFXY::XY,  IFXY::Y2,         /*3*/ IFXY::X3,
    IFXY::X2Y,       IFXY::XY2,        IFXY::Y3,   /*4*/ IFXY::X4,  IFXY::X3Y, IFXY::X2Y2,       IFXY::XY3,
    IFXY::Y4,        /*5*/ IFXY::X4Y,  IFXY::X3Y2, IFXY::X2Y3,      IFXY::XY4, /*6*/ IFXY::X4Y2, IFXY::X3Y3,
    IFXY::X2Y4,      /*7*/ IFXY::X4Y3, IFXY::X3Y4, /*8*/ IFXY::X4Y4};

}  // namespace snapengine
}  // namespace alus
