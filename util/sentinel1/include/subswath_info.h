#pragma once
#include <fstream>
#include <iostream>
#include <vector>
#include "CudaFriendlyObject.h"
#include "cuda_util.hpp"

#include "subswath_info_computation.h"

namespace alus {
namespace s1tbx {

/**
 * This class refers to SubSwathInfo private class from Sentinel1Utils class from s1tbx module
 */
class SubSwathInfo: public cuda::CudaFriendlyObject{
private:
    DeviceSubswathInfo devicePointersHolder;
public:
    //subswath info
    int first_valid_pixel_;
    int last_valid_pixel_;
    double first_line_time_;
    double last_line_time_;
    double first_valid_line_time_;
    double last_valid_line_time_;
    double slr_time_to_first_pixel_;
    double slr_time_to_last_pixel_;
    double slr_time_to_first_valid_pixel_;
    double slr_time_to_last_valid_pixel_;
    double range_pixel_spacing_;
    double azimuth_time_interval_;
    double radar_frequency_;
    double azimuth_steering_rate_;
    std::string subswath_name_;

    // bursts info
    int lines_per_burst_;
    int num_of_bursts_;
    int samples_per_burst_;
    int num_of_samples_;
    int num_of_lines_;
    std::vector<double> burst_first_line_time_; //placeholder
    std::vector<double> burst_last_line_time_;  //placeholder
    //Be careful. The inner vectors can have differing lengths. This is not an orderly matrix
    std::vector<std::vector<int>> first_valid_sample_;
    std::vector<std::vector<int>> last_valid_sample_;
    std::vector<double> burst_first_valid_line_time_;
    std::vector<double> burst_last_valid_line_time_;
    std::vector<int> first_valid_line_;
    std::vector<int> last_valid_line_;

    double **doppler_rate_ = nullptr;
    double **doppler_centroid_ = nullptr;
    double **reference_time_ = nullptr;
    double **range_depend_doppler_rate_ = nullptr;

    // antenna pattern
    // Attention! Inner vectors can have differing lengths. This is not an orderly matrix
    std::vector<std::vector<double>> ap_slant_range_time_;
    std::vector<std::vector<double>> ap_elevation_angle_;

    // GeoLocationGridPoint
    int num_of_geo_lines_;
    int num_of_geo_points_per_line_;
    double **azimuth_time_ = nullptr; //placeholder
    double **slant_range_time_ = nullptr; //placeholder
    double **latitude_ = nullptr;    //placeholder
    double **longitude_ = nullptr;   //placeholder
    double **incidence_angle_ = nullptr; //placeholder

    //the packet that you can use on the gpu
    DeviceSubswathInfo *device_subswath_info_ = nullptr;

    void HostToDevice() override ;
    void DeviceToHost() override ;
    void DeviceFree() override ;

    SubSwathInfo();
    ~SubSwathInfo();
    SubSwathInfo(const SubSwathInfo&) = delete;  // class does not support copying(and moving)
    SubSwathInfo& operator=(const SubSwathInfo&) = delete;

};

}//namespace
}//namespace
