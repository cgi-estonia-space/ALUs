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
#include "snap-core/core/datamodel/metadata_element.h"
#include "point.h"
#include "s1tbx-commons/orbit.h"

namespace alus {
namespace coherence_cuda {
class MetaData {
private:
    s1tbx::Point approx_xyz_centre_original_, approx_radar_centre_original_;
    int band_x_size_, band_x_min_, band_x_max_, band_y_size_, band_y_min_, band_y_max_;
    double t_range_1_, rsr_2_x_, t_azi_1_, line_time_interval_, radar_wavelength_, range_azimuth_spacing_ratio_;
    bool near_range_on_left_;
    std::shared_ptr<s1tbx::Orbit> orbit_;

    static bool IsNearRangeOnLeft(GdalTileReader* incidence_angle_reader);

public:
    MetaData(GdalTileReader* incidence_angle_reader, std::shared_ptr<snapengine::MetadataElement> root_element,
             int orbit_degree);
    MetaData(bool is_near_range_on_left, std::shared_ptr<snapengine::MetadataElement> root_element,
             int orbit_degree);
    // Convert pixel number to range time (1 is first pixel)
    [[nodiscard]] double PixelToTimeRange(double pixel) const;
    double Line2Ta(int line);
    [[nodiscard]] s1tbx::Point GetApproxXyzCentreOriginal();
    [[nodiscard]] s1tbx::Point GetApproxRadarCentreOriginal();
    [[nodiscard]] std::shared_ptr<s1tbx::Orbit> GetOrbit() { return orbit_; }
    [[nodiscard]] int GetBandXSize() const { return band_x_size_; }
    [[nodiscard]] int GetBandYSize() const { return band_y_size_; }
    [[nodiscard]] double GetRadarWaveLength() const { return radar_wavelength_; }
    [[nodiscard]] double GetRangeAzimuthSpacingRatio() const { return range_azimuth_spacing_ratio_; }
};
}  // namespace coherence_cuda
}  // namespace alus
