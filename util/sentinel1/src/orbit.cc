#include "orbit.h"

#include <cmath>

#include "constants.h"
#include "date_utils.h"
#include "meta_data_node_names.h"
#include "orbit_state_vector.h"
#include "poly_utils.h"

namespace alus::s1tbx {
const double Orbit::CRITERPOS = pow(10, -6);
const double Orbit::CRITERTIM = pow(10, -10);

double Orbit::Eq1DopplerDt(Point delta, Point satellite_velocity, Point satellite_acceleration) {
    return satellite_acceleration.In(delta) - satellite_velocity.GetX() * satellite_velocity.GetX() -
           satellite_velocity.GetY() * satellite_velocity.GetY() -
           satellite_velocity.GetZ() * satellite_velocity.GetZ();
}

Point Orbit::GetXyzDotDot(double az_time) {
    // normalize time
    double az_time_normal = (az_time - this->time_[this->time_.size() / 2]) / 10.0;

    double x = 0, y = 0, z = 0;
    for (int i = 2; i <= poly_degree_; ++i) {
        double pow_t = ((i - 1) * i) * pow(az_time_normal, i - 2);
        x += coeff_x_[i] * pow_t;
        y += coeff_y_[i] * pow_t;
        z += coeff_z_[i] * pow_t;
    }

    return Point(x / 100.0, y / 100.0, z / 100.0);
}

Point Orbit::Xyz2T(Point point_on_ellips, double time_azimuth) {
    Point delta;

    int iter;
    double solution;
    for (iter = 0; iter <= maxiter_; ++iter) {
        Point satellite_position = GetXyz(time_azimuth);
        Point satellite_velocity = GetXyzDot(time_azimuth);
        Point satellite_acceleration = GetXyzDotDot(time_azimuth);
        delta = point_on_ellips.Min(satellite_position);

        // update solution
        solution =
            -Eq1Doppler(satellite_velocity, delta) / Eq1DopplerDt(delta, satellite_velocity, satellite_acceleration);
        time_azimuth += solution;

        if (std::abs(solution) < CRITERTIM) {
            break;
        }
    }
    // Compute range time
    // Update equations
    Point satellite_position = GetXyz(time_azimuth);
    delta = point_on_ellips.Min(satellite_position);
    double time_range = delta.Norm() / C;

    // todo: check if 0.0 is ok for z
    return Point(time_range, time_azimuth, 0.0);
}

Point Orbit::RowsColumns2Xyz(int rows, int columns, double az_time, double rg_time, Point &ellipsoid_position) {
    return RowsColumnsHeightToXyz(rows, columns, 0, az_time, rg_time, ellipsoid_position);
}

// PolyUtils.PolyVal1D(azTimeNormal, coeff_X),
// PolyUtils.PolyVal1D(azTimeNormal, coeff_Y),
// PolyUtils.PolyVal1D(azTimeNormal, coeff_Z));
Point Orbit::GetXyz(const double az_time) {
    double az_time_normal = (az_time - this->time_[this->time_.size() / 2]) / 10.0;
    return Point(PolyUtils::PolyVal1D(az_time_normal, this->coeff_x_),
                 PolyUtils::PolyVal1D(az_time_normal, this->coeff_y_),
                 PolyUtils::PolyVal1D(az_time_normal, this->coeff_z_));
}

Point Orbit::GetXyzDot(double az_time) {
    // normalize time
    az_time = (az_time - this->time_[num_state_vectors_ / 2]) / 10.0;
    size_t degree = this->coeff_x_.size() - 1;
    double x = this->coeff_x_.at(1);
    double y = this->coeff_y_.at(1);
    double z = this->coeff_z_.at(1);
    for (size_t i = 2; i <= degree; ++i) {
        double pow_t = i * pow(az_time, i - 1);
        x += this->coeff_x_.at(i) * pow_t;
        y += this->coeff_y_.at(i) * pow_t;
        z += this->coeff_z_.at(i) * pow_t;
    }
    return Point(x / 10.0, y / 10.0, z / 10.0);
}

// eq1_Doppler
double Orbit::Eq1Doppler(Point &sat_velocity, Point &point_on_ellips) { return sat_velocity.In(point_on_ellips); }

// eq2_Range
double Orbit::Eq2Range(Point &point_ellips_sat, double rg_time) {
    // SOL vs C
    return point_ellips_sat.In(point_ellips_sat) - pow(C * rg_time, 2);
}

// eq3_Ellipsoid

double Orbit::Eq3Ellipsoid(const Point &point_on_ellips, double height) {
    return ((point_on_ellips.GetX() * point_on_ellips.GetX() + point_on_ellips.GetY() * point_on_ellips.GetY()) /
            pow(ELL_A + height, 2)) +
           pow(point_on_ellips.GetZ() / (ELL_B + height), 2) - 1.0;
}

// from jlinda: Orbit class//Point lph2xyz(final double line, final double pixel, final double height, final SLCImage
// slcimage)
Point Orbit::RowsColumnsHeightToXyz(
    int rows, int columns, int height, double az_time, double rg_time, Point &ellipsoid_position) {
    (void)rows;
    (void)columns;
    Point satellite_position;
    Point satellite_velocity;

    // allocate matrices
    std::vector<double> equation_set(3);

    std::vector<std::vector<double>> partials_xyz;
    partials_xyz.resize(3, std::vector<double>(3));

    satellite_position = GetXyz(az_time);

    satellite_velocity = GetXyzDot(az_time);

    // iterate for the solution
    for (int iter = 0; iter <= maxiter_; iter++) {
        // update equations and solve system
        Point dsat_p = ellipsoid_position.Min(satellite_position);
        equation_set[0] = -Eq1Doppler(satellite_velocity, dsat_p);
        equation_set[1] = -Eq2Range(dsat_p, rg_time);
        equation_set[2] = -Eq3Ellipsoid(ellipsoid_position, height);
        partials_xyz.at(0).at(0) = satellite_velocity.GetX();
        partials_xyz[0][1] = satellite_velocity.GetY();
        partials_xyz[0][2] = satellite_velocity.GetZ();
        partials_xyz[1][0] = 2 * dsat_p.GetX();
        partials_xyz[1][1] = 2 * dsat_p.GetY();
        partials_xyz[1][2] = 2 * dsat_p.GetZ();
        partials_xyz[2][0] = (2 * ellipsoid_position.GetX()) / (pow(ELL_A + height, 2));
        partials_xyz[2][1] = (2 * ellipsoid_position.GetY()) / (pow(ELL_A + height, 2));
        partials_xyz[2][2] = (2 * ellipsoid_position.GetZ()) / (pow(ELL_B + height, 2));

        // solve system [NOTE!] orbit has to be normalized, otherwise close to singular
        std::vector<double> ellipsoid_position_solution = PolyUtils::Solve33(partials_xyz, equation_set);

        // update solution
        ellipsoid_position.SetX(ellipsoid_position.GetX() + ellipsoid_position_solution[0]);
        ellipsoid_position.SetY(ellipsoid_position.GetY() + ellipsoid_position_solution[1]);
        ellipsoid_position.SetZ(ellipsoid_position.GetZ() + ellipsoid_position_solution[2]);

        // check convergence
        if (std::abs(ellipsoid_position_solution[0]) < CRITERPOS &&
            std::abs(ellipsoid_position_solution[1]) < CRITERPOS &&
            std::abs(ellipsoid_position_solution[2]) < CRITERPOS) {
            break;
        }
    }

    return Point(ellipsoid_position);
}

Orbit::Orbit(snapengine::MetadataElement &element, int degree) {
    std::vector<alus::snapengine::OrbitStateVector> orbit_state_vectors =
        snapengine::MetaDataNodeNames::GetOrbitStateVectors(element);

    num_state_vectors_ = orbit_state_vectors.size();

    time_.resize(num_state_vectors_);
    data_x_.resize(num_state_vectors_);
    data_y_.resize(num_state_vectors_);
    data_z_.resize(num_state_vectors_);

    for (int i = 0; i < num_state_vectors_; i++) {
        // convert time to seconds of the acquisition day
        time_.at(i) = alus::snapengine::DateUtils::DateTimeToSecOfDay(orbit_state_vectors.at(i).time_->ToString());
        data_x_.at(i) = orbit_state_vectors.at(i).x_pos_;
        data_y_.at(i) = orbit_state_vectors.at(i).y_pos_;
        data_z_.at(i) = orbit_state_vectors.at(i).z_pos_;
    }

    poly_degree_ = degree;
    ComputeCoefficients();
}

void Orbit::ComputeCoefficients() {
    coeff_x_ = PolyUtils::PolyFitNormalized(time_, data_x_, poly_degree_);
    coeff_y_ = PolyUtils::PolyFitNormalized(time_, data_y_, poly_degree_);
    coeff_z_ = PolyUtils::PolyFitNormalized(time_, data_z_, poly_degree_);

    is_interpolated_ = true;
}

}  // namespace alus::s1tbx
