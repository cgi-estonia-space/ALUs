#include "processing_velocity_estimate.h"

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>

#include "alus_log.h"
#include "math_utils.h"

using alus::palsar::OrbitInfo;

namespace {
OrbitInfo InterpolateOrbit(const std::vector<OrbitInfo>& osv, double time_point) {
    int n = static_cast<int>(osv.size());
    OrbitInfo r = {};
    r.time_point = time_point;
    for (int i = 0; i < n; i++) {
        double mult = 1;
        for (int j = 0; j < n; j++) {
            if (i == j) continue;

            mult *= (time_point - osv.at(j).time_point) / (osv.at(i).time_point - osv.at(j).time_point);
        }

        r.x_pos += mult * osv[i].x_pos;
        r.y_pos += mult * osv[i].y_pos;
        r.z_pos += mult * osv[i].z_pos;
        r.x_vel += mult * osv[i].x_vel;
        r.y_vel += mult * osv[i].y_vel;
        r.z_vel += mult * osv[i].z_vel;
    }

    return r;
}

namespace WGS84 {                                        // NOLINT
constexpr double A = 6378137.0;                          // m
constexpr double B = 6356752.3142451794975639665996337;  // 6356752.31424518; // m
constexpr double FLAT_EARTH_COEF = 1.0 / ((A - B) / A);  // 298.257223563;
constexpr double E2 = 2.0 / FLAT_EARTH_COEF - 1.0 / (FLAT_EARTH_COEF * FLAT_EARTH_COEF);
}  // namespace WGS84

constexpr double DTOR = M_PI / 180.0;

inline void Geo2xyzWgs84(double latitude, double longitude, double altitude, double& x, double& y, double& z) {
    double const lat = latitude * DTOR;
    double const lon = longitude * DTOR;

    double sin_lat;
    double cos_lat;
    sincos(lat, &sin_lat, &cos_lat);

    double const sinLat = sin_lat;

    double const N = (WGS84::A / sqrt(1.0 - WGS84::E2 * sinLat * sinLat));
    double const NcosLat = (N + altitude) * cos_lat;

    double sin_lon;
    double cos_lon;
    sincos(lon, &sin_lon, &cos_lon);

    x = NcosLat * cos_lon;
    y = NcosLat * sin_lon;
    z = (N + altitude - WGS84::E2 * N) * sinLat;
}

double SquareFitVr(const Eigen::VectorXd& xvals, const Eigen::VectorXd& yvals) {
    Eigen::MatrixXd A(xvals.size(), 2);

    A.setZero();

    for (int i = 0; i < xvals.size(); i++) {
        A(i, 0) = 1.0;
    }

    for (int j = 0; j < xvals.size(); j++) {
        A(j, 1) = xvals(j) * xvals(j);
    }

    auto Q = A.fullPivHouseholderQr();
    auto result = Q.solve(yvals);

    return sqrt(result[1]);
}

double CalcDistance(OrbitInfo pos, double x, double y, double z) {
    double dx = pos.x_pos - x;
    double dy = pos.y_pos - y;
    double dz = pos.z_pos - z;
    return sqrt(dx * dx + dy * dy + dz * dz);
}

}  // namespace

namespace alus::palsar {
double EstimateProcessingVelocity(const SARMetadata& metadata) {
    const double PRI = 1 / metadata.pulse_repetition_frequency;
    const double center_time_point = (metadata.center_time - metadata.first_orbit_time).total_milliseconds() / 1000.0;

    const int center_az_idx = metadata.img.azimuth_size / 2;
    const int center_range_idx = metadata.img.range_size / 2;

    double x, y, z;
    Geo2xyzWgs84(metadata.center_lat, metadata.center_lon, 0, x, y, z);

    auto mid_point = InterpolateOrbit(metadata.orbit_state_vectors, center_time_point);

    double min = CalcDistance(mid_point, x, y, z);
    int min_idx = center_az_idx;
    OrbitInfo min_pos = {};

    // find the closest Orbit state vector to center point from both directions
    for (int i = center_az_idx + 1; i < metadata.img.azimuth_size; i++) {
        const double idx_time = center_time_point + (i - center_az_idx) * PRI;
        auto pos = InterpolateOrbit(metadata.orbit_state_vectors, idx_time);
        const double new_dist = CalcDistance(pos, x, y, z);
        if (new_dist < min) {
            min = new_dist;
            min_idx = i;
            min_pos = pos;
        } else {
            break;
        }
    }

    for (int i = center_az_idx - 1; i > 0; i--) {
        const double idx_time = center_time_point + (i - center_az_idx) * PRI;
        auto pos = InterpolateOrbit(metadata.orbit_state_vectors, idx_time);
        const double new_dist = CalcDistance(pos, x, y, z);
        if (new_dist < min) {
            min = new_dist;
            min_idx = i;
            min_pos = pos;
        } else {
            break;
        }
    }

    LOGD << "Vr estimate min index = " << min_idx << " az mid = " << center_az_idx;
    double R0 = metadata.slant_range_first_sample + (metadata.img.range_size / 2) * metadata.range_spacing;

    const int aperature_size = CalcAperturePixels(metadata, center_range_idx);

    const int N = 9;

    const int step = aperature_size / (N - 1);

    // calculate distance between center point and positisons on the aperture
    // goal is to find data points from real orbit state vector for the hyperbolic range function
    // R^2(n) = R0^2 + Vr^2 * (n)
    // R - slant range across azimuth time points
    // R0 - slant range and closes point
    // Vr - processing / effective radar velocity
    // (n) - relative azimuth time
    Eigen::VectorXd y_vals(N);  // Vr
    Eigen::VectorXd x_vals(N);  // t
    for (int i = 0; i < N; i++) {
        const int neg_idx = (i - (N / 2)) * step;
        const double idx_time = min_pos.time_point + neg_idx * PRI;
        auto pos = InterpolateOrbit(metadata.orbit_state_vectors, idx_time);
        double dx = pos.x_pos - x;
        double dy = pos.y_pos - y;
        double dz = pos.z_pos - z;
        double R_square = dx * dx + dy * dy + dz * dz;
        double R0_square = R0 * R0;
        double dt = min_pos.time_point - pos.time_point;

        y_vals[i] = R_square - R0_square;
        x_vals[i] = dt;
    }

    // Now we have calculated data, where slant range to center point varies with azimuth time

    // mathematically the vectors now contain data points for the equation
    // y = ax^2 + c, best data fit for a means gives us an estimate for Vr^2

    return SquareFitVr(x_vals, y_vals);
}
}  // namespace alus::palsar