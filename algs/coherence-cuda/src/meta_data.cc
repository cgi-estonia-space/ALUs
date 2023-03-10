/**
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
#include "meta_data.h"

#include "alus_log.h"

#include "jlinda/jlinda-core/ellipsoid.h"
#include "jlinda/jlinda-core/geopoint.h"
#include "jlinda/jlinda-core/utils/date_utils.h"
#include "snap-engine-utilities/engine-utilities/datamodel/metadata/abstract_metadata.h"
#include "snap-engine-utilities/engine-utilities/eo/constants.h"


namespace alus {
namespace coherence_cuda {
MetaData::MetaData(bool is_near_range_on_left, std::shared_ptr<snapengine::MetadataElement> element, int orbit_degree,
                   double avg_incidence_angle) {
    this->near_range_on_left_ = is_near_range_on_left;

    // todo: check what snap uses! this is custom solution
    band_x_size_ = element->GetAttributeInt(snapengine::AbstractMetadata::NUM_SAMPLES_PER_LINE);
    band_y_size_ = element->GetAttributeInt(snapengine::AbstractMetadata::NUM_OUTPUT_LINES);

    //  todo:not sure if we still need these 4 below
    //band_x_min_ = 0;
    //band_y_min_ = 0;
    band_x_max_ = band_x_size_ - 1;
    //band_y_max_ = band_y_size_ - 1;

    line_time_interval_ = element->GetAttributeDouble(snapengine::AbstractMetadata::LINE_TIME_INTERVAL);

    constexpr long MEGA{1000000};
    radar_wavelength_ = (snapengine::eo::constants::LIGHT_SPEED / MEGA) /
                        element->GetAttributeDouble(alus::snapengine::AbstractMetadata::RADAR_FREQUENCY);
    t_azi_1_ = s1tbx::DateUtils::DateTimeToSecOfDay(
        element->GetAttributeUtc(alus::snapengine::AbstractMetadata::FIRST_LINE_TIME)->ToString());
    t_range_1_ = element->GetAttributeDouble(alus::snapengine::AbstractMetadata::SLANT_RANGE_TO_FIRST_PIXEL) /
                 snapengine::eo::constants::LIGHT_SPEED;
    rsr_2_x_ = element->GetAttributeDouble(alus::snapengine::AbstractMetadata::RANGE_SAMPLING_RATE) * MEGA * 2;


    auto last_line_time = s1tbx::DateUtils::DateTimeToSecOfDay(
        element->GetAttributeUtc(alus::snapengine::AbstractMetadata::LAST_LINE_TIME)->ToString());

    LOGI << "first line = " << element->GetAttributeUtc(alus::snapengine::AbstractMetadata::FIRST_LINE_TIME)->ToString();

    printf("band y sz = %d\n", band_y_size_);
    printf("taz = central = %.16f first = %.16f  last = %.16f\n", t_azi_1_ + 0.5*band_y_size_ * line_time_interval_, t_azi_1_, last_line_time);
    printf("avg = %.16f\n", (t_azi_1_ + last_line_time) / 2);

    central_avg_az_time = (t_azi_1_ + last_line_time) * 0.5;

    //LOGI << "tazi =  " << t_azi_1_;

    approx_radar_centre_original_.SetX(
        element->GetAttributeDouble(alus::snapengine::AbstractMetadata::NUM_SAMPLES_PER_LINE) /
        2.0);  // x direction is range!
    approx_radar_centre_original_.SetY(
        element->GetAttributeDouble(alus::snapengine::AbstractMetadata::NUM_OUTPUT_LINES) /
        2.0);  // y direction is azimuth
    approx_radar_centre_original_.SetZ(0.0);

    jlinda::GeoPoint approx_geo_centre_original_{};
    approx_geo_centre_original_.lat_ =
        (float)((element->GetAttributeDouble(alus::snapengine::AbstractMetadata::FIRST_NEAR_LAT) +
                 element->GetAttributeDouble(alus::snapengine::AbstractMetadata::FIRST_FAR_LAT) +
                 element->GetAttributeDouble(alus::snapengine::AbstractMetadata::LAST_NEAR_LAT) +
                 element->GetAttributeDouble(alus::snapengine::AbstractMetadata::LAST_FAR_LAT)) /
                4);

    approx_geo_centre_original_.lon_ =
        (float)((element->GetAttributeDouble(alus::snapengine::AbstractMetadata::FIRST_NEAR_LONG) +
                 element->GetAttributeDouble(alus::snapengine::AbstractMetadata::FIRST_FAR_LONG) +
                 element->GetAttributeDouble(alus::snapengine::AbstractMetadata::LAST_NEAR_LONG) +
                 element->GetAttributeDouble(alus::snapengine::AbstractMetadata::LAST_FAR_LONG)) /
                4);

    //    todo: refactor...
    std::vector<double> xyz(3);
    alus::jlinda::Ellipsoid::Ell2Xyz(approx_geo_centre_original_, xyz);
    approx_xyz_centre_original_ = s1tbx::Point(xyz.at(0), xyz.at(1), xyz.at(2));

    orbit_ = std::make_shared<s1tbx::Orbit>(element, orbit_degree);

    double range_spacing = element->GetAttributeDouble(alus::snapengine::AbstractMetadata::RANGE_SPACING);

    range_spacing /= sin(avg_incidence_angle * alus::snapengine::eo::constants::DTOR);

    ground_range_azimuth_spacing_ratio_ =
        range_spacing / element->GetAttributeDouble(alus::snapengine::AbstractMetadata::AZIMUTH_SPACING);
}

std::vector<BurstData> FillBurstInfo(s1tbx::Sentinel1Utils* su)
{
    auto* swath = su->subswath_[0].get();
    std::vector<BurstData> vec;

    for(int i = 0; i < swath->num_of_bursts_; i++)
    {

        BurstData bd = {};
        bd.first_line_time = swath->burst_first_line_time_.at(i);
        bd.last_line_time = swath->burst_last_line_time_.at(i);
        auto slr_time_first_pixel = swath->slr_time_to_first_pixel_;
        auto slr_time_last_pixel = swath->slr_time_to_last_pixel_;
        const double latUL = su->GetLatitude(bd.first_line_time, slr_time_first_pixel, swath);
        const double latUR = su->GetLatitude(bd.first_line_time, slr_time_last_pixel, swath);
        const double latLL = su->GetLatitude(bd.last_line_time, slr_time_first_pixel, swath);
        const double latLR = su->GetLatitude(bd.last_line_time, slr_time_first_pixel, swath);

        const double lonUL = su->GetLongitude(bd.first_line_time, slr_time_first_pixel, swath);
        const double lonUR = su->GetLongitude(bd.first_line_time, slr_time_last_pixel, swath);
        const double lonLL = su->GetLongitude(bd.last_line_time, slr_time_first_pixel, swath);
        const double lonLR = su->GetLongitude(bd.last_line_time, slr_time_last_pixel, swath);

        jlinda::GeoPoint approx_geo_centre{};
        approx_geo_centre.lat_ = (latUL + latUR + latLL + latLR)/4.0;
        approx_geo_centre.lon_ = (lonUL + lonUR + lonLL + lonLR)/4.0;
        std::vector<double> xyz(3);
        alus::jlinda::Ellipsoid::Ell2Xyz(approx_geo_centre, xyz);
        bd.approx_xyz_centre = s1tbx::Point(xyz.at(0), xyz.at(1), xyz.at(2));
        vec.push_back(bd);
    }
    return vec;
}

// pix2tr
// https://github.com/senbox-org/s1tbx/blob/master/jlinda/jlinda-core/src/main/java/org/jlinda/core/SLCImage.java
double MetaData::PixelToTimeRange(double pixel) const {
    if (near_range_on_left_) {
        return t_range_1_ + pixel / rsr_2_x_;
    } else {
        return t_range_1_ + (band_x_max_ - 1 - pixel) / rsr_2_x_;
    }
}

//s1tbx::Point MetaData::GetApproxXyzCentreOriginal() { return s1tbx::Point(approx_xyz_centre_original_); }

// original input was double...
double MetaData::Line2Ta(int burst_index, int line) {
    double first_line_in_days = burst_meta.at(burst_index).first_line_time / 86400;
    double first_line_time = (first_line_in_days - (int) first_line_in_days) * 86400;
    return first_line_time + line * line_time_interval_;
}

//s1tbx::Point MetaData::GetApproxRadarCentreOriginal() { return approx_radar_centre_original_; }
}  // namespace coherence-cuda
}  // namespace alus
