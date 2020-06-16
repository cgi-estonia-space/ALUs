#pragma once

#include <vector>

#include "constants.h"
#include "meta_data.h"
#include "point.h"
#include "utils.h"

namespace alus {
class Orbit {
   private:
    std::vector<double> coeff_x_, coeff_y_, coeff_z_, time_;
    static const int maxiter_ = 10;
    int orbit_degree_, poly_degree_, num_state_vectors_;
    MetaData &metadata_;
    // might need to move this to header
    static const double CRITERPOS_;
    static const double CRITERTIM_;

    Point RowsColumnsHeightToXyz(int rows, int columns, int height, MetaData &meta_data);

   public:
    Orbit(MetaData &meta_data, int orbit_degree);
    // pixels to coordinates
    Point RowsColumns2Xyz(int rows, int columns);
    Point Xyz2T(Point point_on_ellips, MetaData slave_meta_data);
    [[nodiscard]] Point GetXyz(double az_time);
    [[nodiscard]] Point GetXyzDot(double az_time);
    [[nodiscard]] Point GetXyzDotDot(double az_time);
    double Eq1Doppler(Point &sat_velocity, Point &point_on_ellips);
    double Eq2Range(Point &point_ellips_sat, double rg_time);
    double Eq3Ellipsoid(Point &point_on_ellips, double height);
    double Eq1DopplerDt(Point delta, Point satellite_velocity, Point satellite_acceleration);
};
}  // namespace alus
