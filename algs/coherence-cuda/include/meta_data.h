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
#pragma once

#include <memory>

#include "gdal_tile_reader.h"
#include "point.h"
#include "s1tbx-commons/orbit.h"
#include "snap-core/core/datamodel/metadata_element.h"
#include "s1tbx-commons/sentinel1_utils.h"

namespace alus {
namespace coherence_cuda {


struct BurstData
{
    double first_line_time;
    double last_line_time;
    s1tbx::Point approx_xyz_centre;
};


std::vector<BurstData> FillBurstInfo(s1tbx::Sentinel1Utils* su);

class MetaData {

public:
    std::vector<BurstData> burst_meta;
    int lines_per_burst;
public:
    s1tbx::Point approx_xyz_centre_original_;
    s1tbx::Point approx_radar_centre_original_;
    int band_x_size_;
    //int band_x_min_;
    int band_x_max_;
    int band_y_size_;
    //int band_y_min_;
    //int band_y_max_;
    double t_range_1_;
    double rsr_2_x_;
    double t_azi_1_;
    double line_time_interval_;
    double radar_wavelength_;
    double ground_range_azimuth_spacing_ratio_;
    double central_avg_az_time;
    bool near_range_on_left_;
    std::shared_ptr<s1tbx::Orbit> orbit_;

public:
    MetaData(bool is_near_range_on_left, std::shared_ptr<snapengine::MetadataElement> root_element, int orbit_degree,
             double avg_incidence_angle);
    // Convert pixel number to range time (1 is first pixel)
    [[nodiscard]] double PixelToTimeRange(double pixel) const;
    double Line2Ta(int burst_index, int line);
    //[[nodiscard]] s1tbx::Point GetApproxXyzCentreOriginal();
    //[[nodiscard]] s1tbx::Point GetApproxRadarCentreOriginal();
    [[nodiscard]] std::shared_ptr<s1tbx::Orbit> GetOrbit() { return orbit_; }
    [[nodiscard]] int GetBandXSize() const { return band_x_size_; }
    [[nodiscard]] int GetBandYSize() const { return band_y_size_; }
    [[nodiscard]] double GetRadarWaveLength() const { return radar_wavelength_; }
    [[nodiscard]] double GetRangeAzimuthSpacingRatio() const { return ground_range_azimuth_spacing_ratio_; }
};
}  // namespace coherence-cuda
}  // namespace alus
