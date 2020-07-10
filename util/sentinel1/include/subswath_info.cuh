#pragma once

namespace alus {

struct DeviceSubswathInfo{
    //subswath info
    int first_valid_pixel;
    int last_valid_pixel;
    double first_line_time;
    double last_line_time;
    double slr_time_to_first_pixel;
    double slr_time_to_last_pixel;
    double range_pixel_spacing;
    double azimuth_time_interval;
    double radar_frequency;
    double azimuth_steering_rate;

    double *device_burst_first_line_time = nullptr;
    double *device_burst_last_line_time = nullptr;

    double *device_doppler_rate = nullptr;
    double *device_doppler_centroid = nullptr;
    double *device_reference_time = nullptr;
    double *device_range_depend_doppler_rate = nullptr;

    int doppler_size_x, doppler_size_y;

    // bursts info
    int lines_per_burst;
    int num_of_bursts;
    int samples_per_burst;

    // GeoLocationGridPoint
    int num_of_geo_lines;
    int num_of_geo_points_per_line;

};

}//namespace
