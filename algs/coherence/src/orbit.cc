#include "orbit.h"

namespace alus {
const double Orbit::CRITERPOS_ = pow(10, -6);
const double Orbit::CRITERTIM_ = pow(10, -10);

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

Point Orbit::Xyz2T(Point point_on_ellips, MetaData slave_meta_data) {
    Point delta;

    // inital value
    double time_azimuth =
        slave_meta_data.Line2Ta(static_cast<int>(0.5 * slave_meta_data.GetApproxRadarCentreOriginal().GetY()));

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

        if (std::abs(solution) < CRITERTIM_) {
            break;
        }
    }
    // Compute range time
    // Update equations
    Point satellite_position = GetXyz(time_azimuth);
    delta = point_on_ellips.Min(satellite_position);
    double time_range = delta.Norm() / alus::kcoh::C;

    // todo: check if 0.0 is ok for z
    return Point(time_range, time_azimuth, 0.0);
}

Orbit::Orbit(MetaData &meta_data, int orbit_degree) : metadata_(meta_data) {
    //    this->metadata_ = meta_data;
    this->orbit_degree_ = orbit_degree;
    // slave
    this->poly_degree_ = 3;
    this->num_state_vectors_ = 23;

    //    todo:check if this is correct
    // master
    this->time_ = meta_data.time_;

    this->coeff_x_ = meta_data.coeff_x_;
    this->coeff_y_ = meta_data.coeff_y_;
    this->coeff_z_ = meta_data.coeff_z_;
}

Point Orbit::RowsColumns2Xyz(int rows, int columns) {
    return RowsColumnsHeightToXyz(rows, columns, 0, this->metadata_);
}

// PolyUtils.PolyVal1D(azTimeNormal, coeff_X),
// PolyUtils.PolyVal1D(azTimeNormal, coeff_Y),
// PolyUtils.PolyVal1D(azTimeNormal, coeff_Z));
Point Orbit::GetXyz(const double az_time) {
    double az_time_normal = (az_time - this->time_[this->time_.size() / 2]) / 10.0;
    return Point(Utils::PolyVal1D(az_time_normal, this->coeff_x_),
                 Utils::PolyVal1D(az_time_normal, this->coeff_y_),
                 Utils::PolyVal1D(az_time_normal, this->coeff_z_));
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
    return point_ellips_sat.In(point_ellips_sat) - pow(alus::kcoh::C * rg_time, 2);
}

// eq3_Ellipsoid

double Orbit::Eq3Ellipsoid(Point &point_on_ellips, double height) {
    return ((point_on_ellips.GetX() * point_on_ellips.GetX() + point_on_ellips.GetY() * point_on_ellips.GetY()) /
            pow(alus::kcoh::ELL_A + height, 2)) +
           pow(point_on_ellips.GetZ() / (alus::kcoh::ELL_B + height), 2) - 1.0;
}

// from jlinda: Orbit class//Point lph2xyz(final double line, final double pixel, final double height, final SLCImage
// slcimage)
Point Orbit::RowsColumnsHeightToXyz(int rows, int columns, int height, MetaData &meta_data) {
    Point satellite_position;
    Point satellite_velocity;
    Point ellipsoid_position;  // returned

    // allocate matrices
    std::vector<double> equation_set(3);

    std::vector<std::vector<double>> partials_xyz;
    partials_xyz.resize(3, std::vector<double>(3));

    double az_time = meta_data.Line2Ta(rows);
    double rg_time = meta_data.PixelToTimeRange(columns);

    satellite_position = GetXyz(az_time);

    satellite_velocity = GetXyzDot(az_time);

    // initial value
    ellipsoid_position = meta_data.GetApproxXyzCentreOriginal();

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
        partials_xyz[2][0] = (2 * ellipsoid_position.GetX()) / (pow(alus::kcoh::ELL_A + height, 2));
        partials_xyz[2][1] = (2 * ellipsoid_position.GetY()) / (pow(alus::kcoh::ELL_A + height, 2));
        partials_xyz[2][2] = (2 * ellipsoid_position.GetZ()) / (pow(alus::kcoh::ELL_B + height, 2));

        // solve system [NOTE!] orbit has to be normalized, otherwise close to singular
        std::vector<double> ellipsoid_position_solution = Utils::Solve33(partials_xyz, equation_set);

        // update solution
        ellipsoid_position.SetX(ellipsoid_position.GetX() + ellipsoid_position_solution[0]);
        ellipsoid_position.SetY(ellipsoid_position.GetY() + ellipsoid_position_solution[1]);
        ellipsoid_position.SetZ(ellipsoid_position.GetZ() + ellipsoid_position_solution[2]);

        // check convergence
        if (std::abs(ellipsoid_position_solution[0]) < CRITERPOS_ &&
            std::abs(ellipsoid_position_solution[1]) < CRITERPOS_ &&
            std::abs(ellipsoid_position_solution[2]) < CRITERPOS_) {
            break;
        }
    }

    return Point(ellipsoid_position);
}

}  // namespace alus
