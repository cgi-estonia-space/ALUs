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

#include "date_utils.h"
#include "jlinda-core/constants.h"
#include "jlinda-core/ellipsoid.h"
#include "jlinda-core/geopoint.h"
#include "snap-engine-utilities/datamodel/metadata/abstract_metadata.h"

namespace alus {

MetaData::MetaData(IDataTileReader* incidence_angle_reader, std::shared_ptr<snapengine::MetadataElement> element,
                   int orbit_degree) : MetaData(IsNearRangeOnLeft(incidence_angle_reader), element, orbit_degree){
}

MetaData::MetaData(bool is_near_range_on_left, std::shared_ptr<snapengine::MetadataElement> element, int orbit_degree) {

    this->near_range_on_left_ = is_near_range_on_left;

    // todo: check what snap uses! this is custom solution
    band_x_size_ = element->GetAttributeInt(snapengine::AbstractMetadata::NUM_SAMPLES_PER_LINE);
    band_y_size_ = element->GetAttributeInt(snapengine::AbstractMetadata::NUM_OUTPUT_LINES);

    //  todo:not sure if we still need these 4 below
    band_x_min_ = 0;
    band_y_min_ = 0;
    band_x_max_ = band_x_size_ - 1;
    band_y_max_ = band_y_size_ - 1;

    line_time_interval_ = element->GetAttributeDouble(snapengine::AbstractMetadata::LINE_TIME_INTERVAL);

    radar_wavelength_ = (jlinda::LIGHT_SPEED / jlinda::MEGA) /
                        element->GetAttributeDouble(alus::snapengine::AbstractMetadata::RADAR_FREQUENCY);
    t_azi_1_ = snapengine::DateUtils::DateTimeToSecOfDay(
        element->GetAttributeUtc(alus::snapengine::AbstractMetadata::FIRST_LINE_TIME)->ToString());
    t_range_1_ = element->GetAttributeDouble(alus::snapengine::AbstractMetadata::SLANT_RANGE_TO_FIRST_PIXEL) /
                 jlinda::LIGHT_SPEED;
    rsr_2_x_ = element->GetAttributeDouble(alus::snapengine::AbstractMetadata::RANGE_SAMPLING_RATE) * jlinda::MEGA * 2;

    approx_radar_centre_original_.SetX(static_cast<double>(band_x_size_) / 2.0);  // x direction is range!
    approx_radar_centre_original_.SetY(static_cast<double>(band_y_size_) / 2.0);  // y direction is azimuth
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

}

// todo: how should we tie this to specific product in our logic?
bool MetaData::IsNearRangeOnLeft(IDataTileReader* incidence_angle_reader) {
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

s1tbx::Point MetaData::GetApproxXyzCentreOriginal() { return s1tbx::Point(this->approx_xyz_centre_original_); }

// original input was double...
double MetaData::Line2Ta(int line) {
    return this->t_azi_1_ + (line - 1) * this->line_time_interval_;
    //        return t_azi_1_ + ((line - 1.0) / PRF);
}

s1tbx::Point MetaData::GetApproxRadarCentreOriginal() { return this->approx_radar_centre_original_; }
}  // namespace alus
