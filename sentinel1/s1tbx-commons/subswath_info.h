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

#include <fstream>
#include <vector>

#include "cuda_friendly_object.h"
#include "cuda_util.h"

#include "subswath_info_computation.h"

namespace alus {
namespace s1tbx {

/**
 * This class refers to SubSwathInfo private class from Sentinel1Utils class from s1tbx module
 */
class SubSwathInfo : public cuda::CudaFriendlyObject {
public:
    DeviceSubswathInfo devicePointersHolder = {};
    // subswath info
    int first_valid_pixel_ = 0;
    int last_valid_pixel_ = 0;
    double first_line_time_ = 0.0;
    double last_line_time_ = 0.0;
    double first_valid_line_time_ = 0.0;
    double last_valid_line_time_ = 0.0;
    double slr_time_to_first_pixel_ = 0.0;
    double slr_time_to_last_pixel_ = 0.0;
    double slr_time_to_first_valid_pixel_ = 0.0;
    double slr_time_to_last_valid_pixel_ = 0.0;
    double range_pixel_spacing_ = 0.0;
    double range_sampling_rate_ = 0.0;
    double azimuth_pixel_spacing_ = 0.0;
    double azimuth_time_interval_ = 0.0;
    double radar_frequency_ = 0.0;
    double azimuth_steering_rate_ = 0.0;
    double ascending_node_time_ = 0.0;
    std::string subswath_name_;

    // bursts info
    int lines_per_burst_ = 0;
    int num_of_bursts_ = 0;
    int samples_per_burst_ = 0;
    int num_of_samples_ = 0;
    int num_of_lines_ = 0;
    std::vector<double> burst_first_line_time_;  // placeholder
    std::vector<double> burst_last_line_time_;   // placeholder
    // Be careful. The inner vectors can have differing lengths. This is not an orderly matrix
    std::vector<std::vector<int>> first_valid_sample_;
    std::vector<std::vector<int>> last_valid_sample_;
    std::vector<double> burst_first_valid_line_time_;
    std::vector<double> burst_last_valid_line_time_;
    std::vector<int> first_valid_line_;
    std::vector<int> last_valid_line_;

    double** doppler_rate_ = nullptr;
    double** doppler_centroid_ = nullptr;
    double** reference_time_ = nullptr;
    double** range_depend_doppler_rate_ = nullptr;

    // antenna pattern
    // Attention! Inner vectors can have differing lengths. This is not an orderly matrix
    std::vector<std::vector<double>> ap_slant_range_time_;
    std::vector<std::vector<double>> ap_elevation_angle_;

    // GeoLocationGridPoint
    int num_of_geo_lines_ = 0;
    int num_of_geo_points_per_line_ = 0;
    double** azimuth_time_ = nullptr;      // placeholder
    double** slant_range_time_ = nullptr;  // placeholder
    double** latitude_ = nullptr;          // placeholder
    double** longitude_ = nullptr;         // placeholder
    double** incidence_angle_ = nullptr;   // placeholder

    // the packet that you can use on the gpu
    DeviceSubswathInfo* device_subswath_info_ = nullptr;

    void HostToDevice() override;
    void DeviceToHost() override;
    void DeviceFree() override;

    double calc_avg_incidence_angle()
    {
        return incidence_angle_[num_of_geo_lines_/2][num_of_geo_points_per_line_/2];
    }

    bool isNearRangeOnLeft()
    {
        double angle_first_pixel = incidence_angle_[0][0];
        double angle_last_pixel = incidence_angle_[0][num_of_geo_points_per_line_-1];
        return angle_first_pixel < angle_last_pixel;
    }

    SubSwathInfo();
    ~SubSwathInfo();
    SubSwathInfo(const SubSwathInfo&) = delete;  // class does not support copying(and moving)
    SubSwathInfo& operator=(const SubSwathInfo&) = delete;

};

}  // namespace s1tbx
}  // namespace alus
