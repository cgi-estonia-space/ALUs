#include "meta_data.h"

namespace alus {
MetaData::MetaData(IDataTileReader *incidence_angle_reader,
                   double t_azi_1,
                   std::vector<double> time,
                   std::vector<double> coeff_x,
                   std::vector<double> coeff_y,
                   std::vector<double> coeff_z) {
    // window in jLinda, could read these using gdal
    //  todo:fix this
    this->band_x_min_ = 0;
    this->band_x_max_ = 21399;
    this->band_x_size_ = 21400;

    this->band_y_min_ = 0;
    this->band_y_max_ = 1502;
    this->band_y_size_ = 1503;

    // used debugger for that (same for slave master at least for me)
    this->line_time_interval_ = 0.002055556299999998;
    this->t_azi_1_ = t_azi_1;

    // todo THESE COME FROM METADATA STATEVECTORS
    this->time_ = time;
    this->coeff_x_ = coeff_x;
    this->coeff_y_ = coeff_y;
    this->coeff_z_ = coeff_z;

    // todo: slave and master need different values hardcoded
    // THIS WOULD BE SLAVE VALUE
    // this->t_azi_1 = 56909.695959
    this->t_range_1_ = 0.002679737321566982;  // print(masterContainer.metaData.gettRange1())
    // 1.2869047625142854E8
    this->rsr_2_x_ = 128690476.25142854;  // took from python print(masterContainer.metaData.getRsr2x())

    // get angle for first and last to understand if this is ok (need to calculate from input product...
    this->near_range_on_left_ = IsNearRangeOnLeft(incidence_angle_reader);

    // todo:where this gets populated?
    this->approx_xyz_centre_original_ = Point(2912246.953674266, 1455420.6100392228, 5466244.152426148);

    // for slave and master these matched
    this->approx_radar_centre_original_ = Point(10700.0, 751.5, 0.0);
}

// todo: how should we tie this to specific product in our logic?
bool MetaData::IsNearRangeOnLeft(IDataTileReader *incidence_angle_reader) {
    const double INCIDENCE_ANGLE_TO_FIRST_PIXEL = incidence_angle_reader->GetValueAtXy(
        incidence_angle_reader->GetBandXMin(), incidence_angle_reader->GetBandYMin());
    const double INCIDENCE_ANGLE_TO_LAST_PIXEL = incidence_angle_reader->GetValueAtXy(
        incidence_angle_reader->GetBandXSize() - 1, incidence_angle_reader->GetBandYMin());
    if (INCIDENCE_ANGLE_TO_FIRST_PIXEL && INCIDENCE_ANGLE_TO_LAST_PIXEL) {
        return (INCIDENCE_ANGLE_TO_FIRST_PIXEL < INCIDENCE_ANGLE_TO_LAST_PIXEL);
    }
    return true;
}

// pix2tr
// https://github.com/senbox-org/s1tbx/blob/master/jlinda/jlinda-core/src/main/java/org/jlinda/core/SLCImage.java
double MetaData::PixelToTimeRange(double pixel) const {
    if (this->near_range_on_left_) {
        return t_range_1_ + pixel / rsr_2_x_;
    } else {
        return t_range_1_ + (this->band_x_max_ - 1 - pixel) / this->rsr_2_x_;
    }
}

Point MetaData::GetApproxXyzCentreOriginal() { return Point(this->approx_xyz_centre_original_); }

// original input was double...
double MetaData::Line2Ta(int line) {
    return this->t_azi_1_ + (line - 1) * this->line_time_interval_;
    //        return t_azi_1_ + ((line - 1.0) / PRF);
}

Point MetaData::GetApproxRadarCentreOriginal() { return this->approx_radar_centre_original_; }
}  // namespace alus
