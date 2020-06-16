#pragma once

#include "i_data_tile_reader.h"
#include "point.h"

namespace alus {
class MetaData {
   private:
    // todo:maybe not needed as a member
    IDataTileReader *incidence_angle_reader_;
    Point approx_xyz_centre_original_, approx_radar_centre_original_;
    int band_x_size_, band_x_min_, band_x_max_, band_y_size_, band_y_min_, band_y_max_;
    double t_range_1_, rsr_2_x_, t_azi_1_, line_time_interval_;
    bool near_range_on_left_;

    bool IsNearRangeOnLeft(IDataTileReader *incidence_angle_reader);

   public:
    // TODO:NEED TO READ FROM META (state vectors)
    //    TODO:why are these public?
    std::vector<double> time_, coeff_x_, coeff_y_, coeff_z_;

    MetaData(IDataTileReader *incidence_angle_reader,
             double t_azi_1,
             std::vector<double> time,
             std::vector<double> coeff_x,
             std::vector<double> coeff_y,
             std::vector<double> coeff_z);
    // Convert pixel number to range time (1 is first pixel)
    double PixelToTimeRange(double pixel) const;
    double Line2Ta(int line);
    [[nodiscard]] Point GetApproxXyzCentreOriginal();
    [[nodiscard]] Point GetApproxRadarCentreOriginal();
    // incident_angle.hdr
};
}  // namespace alus
