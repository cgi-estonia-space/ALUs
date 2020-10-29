#pragma once

#include <vector>

#include <eigen3/Eigen/Dense>

namespace alus {

class PolyUtils {
   public:
    [[nodiscard]] static double PolyVal1D(double x, std::vector<double> coeffs);
    [[nodiscard]] static std::vector<double> Solve33(std::vector<std::vector<double>> a, std::vector<double> rhs);

    static Eigen::VectorXd PolyFit(Eigen::VectorXd xvals, Eigen::VectorXd yvals,int order);

    static Eigen::VectorXd Normalize(Eigen::VectorXd t);

    /**
     * polyfit
     * <p/>
     * Compute coefficients of x=a0+a1*t+a2*t^2+a3*t3 polynomial
     * for orbit interpolation.  Do this to facilitate a method
     * in case only a few datapoints are given.
     * Data t is normalized approximately [-x,x], then polynomial
     * coefficients are computed.  For poly_val this is repeated
     * see getxyz, etc.
     * <p/>
     * input:
     * - matrix by getdata with time and position info
     * output:
     * - matrix with coeff.
     * (input for interp. routines)
     */
    static std::vector<double> PolyFitNormalized(std::vector<double> t,  std::vector<double> y, int degree);

    static Eigen::VectorXd PolyFitNormalized(Eigen::VectorXd t,  Eigen::VectorXd y, int degree);
};



}  // namespace alus
