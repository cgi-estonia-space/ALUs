#include "ellipsoid.h"

#include <cmath>

#include "tensorflow/core/platform/default/logging.h"

#include "constants.h"

namespace alus {
namespace jlinda {

double Ellipsoid::e2_ = 0.00669438003551279091;
double Ellipsoid::e2b_ = 0.00673949678826153145;

double Ellipsoid::a_ = kcoh::WGS84_A;
double Ellipsoid::b_ = kcoh::WGS84_A;
std::string Ellipsoid::name_ = "WGS84";

double Ellipsoid::ComputeEllipsoidNormal(const double phi) { return a_ / sqrt(1.0 - e2_ * pow(sin(phi), 2)); }
double Ellipsoid::ComputeCurvatureRadiusInMeridianPlane(const double phi) {
    return a_ * (1 - e2_) / pow((1 - e2_ * pow(sin(phi), 2)), 3 / 2);
}
void Ellipsoid::SetEcc1stSqr() { e2_ = 1.0 - pow(b_ / a_, 2); }
void Ellipsoid::SetEcc2ndSqr() { e2b_ = pow(a_ / b_, 2) - 1.0; }

Ellipsoid::Ellipsoid() {
    Ellipsoid::a_ = kcoh::WGS84_A;
    Ellipsoid::b_ = kcoh::WGS84_B;
    Ellipsoid::e2_ = 0.00669438003551279091;
    Ellipsoid::e2b_ = 0.00673949678826153145;
    Ellipsoid::name_ = "WGS84";
}

Ellipsoid::Ellipsoid(const double semi_major, const double semi_minor) {
    a_ = semi_major;
    b_ = semi_minor;
    SetEcc1stSqr();  // compute e2 (not required for zero-doppler iter.)
    SetEcc2ndSqr();  // compute e2b (not required for zero-doppler iter.)
}

Ellipsoid::Ellipsoid(const Ellipsoid& ell) {
    a_ = ell.a_;
    b_ = ell.b_;
    e2_ = ell.e2_;
    e2b_ = ell.e2b_;
    name_ = ell.name_;
}

void Ellipsoid::ShowData() {
    LOG(INFO) << "ELLIPSOID: \tEllipsoid used (orbit, output): " + name_ + ".";
    LOG(INFO) << "ELLIPSOID: a   = " + std::to_string(a_);
    LOG(INFO) << "ELLIPSOID: b   = " + std::to_string(b_);
    LOG(INFO) << "ELLIPSOID: e2  = " + std::to_string(e2_);
    LOG(INFO) << "ELLIPSOID: e2' = " + std::to_string(e2b_);
}

std::vector<double> Ellipsoid::Xyz2Ell(const s1tbx::Point& xyz) {
    const double r = sqrt(xyz.GetX() * xyz.GetX() + xyz.GetY() * xyz.GetY());
    const double nu = atan2((xyz.GetZ() * a_), (r * b_));
    const double sinNu = sin(nu);
    const double cosNu = cos(nu);
    const double sin3 = sinNu * sinNu * sinNu;
    const double cos3 = cosNu * cosNu * cosNu;
    const double phi = atan2((xyz.GetZ() + e2b_ * b_ * sin3), (r - e2_ * a_ * cos3));
    const double lambda = atan2(xyz.GetY(), xyz.GetX());
    const double N = ComputeEllipsoidNormal(phi);
    const double height = (r / cos(phi)) - N;
    return std::vector<double>{phi, lambda, height};
}
s1tbx::Point Ellipsoid::Ell2Xyz(const double phi, const double lambda, const double height) {
    if (phi > kcoh::SNAP_PI || phi < -kcoh::SNAP_PI || lambda > kcoh::SNAP_PI || lambda < -kcoh::SNAP_PI) {
        throw std::invalid_argument("Ellipsoid.ell2xyz : input values for phi/lambda have to be in radians!");
    }
    const double N = ComputeEllipsoidNormal(phi);
    const double Nph = N + height;
    const double A = Nph * cos(phi);
    return s1tbx::Point(A * cos(lambda), A * sin(lambda), (Nph - e2_ * N) * sin(phi));
}
s1tbx::Point Ellipsoid::Ell2Xyz(std::vector<double> phi_lambda_height) {
    const double phi = phi_lambda_height.at(0);
    const double lambda = phi_lambda_height.at(1);
    const double height = phi_lambda_height.at(2);

    if (phi > kcoh::SNAP_PI || phi < -kcoh::SNAP_PI || lambda > kcoh::SNAP_PI || lambda < -kcoh::SNAP_PI) {
        throw std::invalid_argument("Ellipsoid.ell2xyz(): phi/lambda values has to be in radians!");
    }

    const double N = ComputeEllipsoidNormal(phi);
    const double Nph = N + height;
    const double A = Nph * cos(phi);
    return s1tbx::Point(A * cos(lambda), A * sin(lambda), (Nph - e2_ * N) * sin(phi));
}

s1tbx::Point Ellipsoid::Ell2Xyz(const GeoPoint& geo_point, const double height) {
    return Ell2Xyz(geo_point.lat_ * kcoh::DTOR, geo_point.lon_ * kcoh::DTOR, height);
}
s1tbx::Point Ellipsoid::Ell2Xyz(const GeoPoint& geo_point) {
    return Ell2Xyz(geo_point.lat_ * kcoh::DTOR, geo_point.lon_ * kcoh::DTOR, 0.0);
}

void Ellipsoid::Ell2Xyz(const GeoPoint& geo_point, std::vector<double>& xyz) {
    s1tbx::Point tempPoint = Ell2Xyz(geo_point.lat_ * kcoh::DTOR, geo_point.lon_ * kcoh::DTOR, 0.0);
    xyz.at(0) = tempPoint.GetX();
    xyz.at(1) = tempPoint.GetY();
    xyz.at(2) = tempPoint.GetZ();
}

void Ellipsoid::Ell2Xyz(const GeoPoint& geo_point, const double height, std::vector<double>& xyz) {
    s1tbx::Point temp_point = Ell2Xyz(geo_point.lat_ * kcoh::DTOR, geo_point.lon_ * kcoh::DTOR, height);
    xyz.at(0) = temp_point.GetX();
    xyz.at(1) = temp_point.GetY();
    xyz.at(2) = temp_point.GetZ();
}
}  // namespace jlinda
}  // namespace alus