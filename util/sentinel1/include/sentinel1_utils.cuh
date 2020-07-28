#pragma once

namespace alus {
namespace s1tbx {

struct DeviceSentinel1Utils {
    double first_line_utc;
    double last_line_utc;
    double line_time_interval;
    double near_edge_slant_range;
    double wavelength;
    double range_spacing;
    double azimuth_spacing;

    int source_image_width;
    int source_image_height;
    int near_range_on_left;
    int srgr_flag;
};

}//namespace
}//namespace