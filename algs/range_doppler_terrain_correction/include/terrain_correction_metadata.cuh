#pragma once

#include "terrain_correction_metadata.h"

namespace alus {
namespace terraincorrection {

struct RangeDopplerKernelMetadata {
    ProductType product_type;
    AcquisitionMode acquisition_mode;
    AntennaDirection antenna_pointing;
    alus::terraincorrection::Swath swath;
    alus::snapengine::Utc proc_time;
    unsigned int orbit_cycle;
    unsigned int rel_orbit;
    unsigned int abs_orbit;
    alus::snapengine::Utc state_vector_time;
    double incidence_near;
    double incidence_far;
    int slice_num;
    int data_take_id;
    alus::snapengine::Utc first_line_time;
    alus::snapengine::Utc last_line_time;
    double first_near_lat;
    double first_near_long;
    double first_far_lat;
    double first_far_long;
    double last_near_lat;
    double last_near_long;
    double last_far_lat;
    double last_far_long;
    alus::terraincorrection::Pass pass;
    alus::terraincorrection::SampleType sample_type;
    alus::terraincorrection::Polarisation mds1_tx_rx_polar;
    alus::terraincorrection::Polarisation mds2_tx_rx_polar;
    alus::terraincorrection::Polarisation mds3_tx_rx_polar;
    alus::terraincorrection::Polarisation mds4_tx_rx_polar;
    alus::terraincorrection::Algorithm algorithm;
    double azimuth_looks;
    double range_looks;
    double range_spacing;
    double azimuth_spacing;
    double pulse_repetition_frequency;
    double radar_frequency;
    double line_time_interval;
    unsigned int total_size;
    unsigned int num_output_lines;
    unsigned int num_samples_per_line;
    unsigned int subset_offset_x;
    unsigned int subset_offset_y;
    bool srgr_flag;
    double avg_scene_height;
    bool is_terrain_corrected;
    double lat_pixel_res;
    double long_pixel_res;
    double slant_range_to_first_pixel;
    bool ant_elev_corr_flag;
    bool range_spread_comp_flag;
    bool replica_power_corr_flag;
    bool abs_calibration_flag;
    double calibration_factor;
    double chirp_power;
    bool inc_angle_comp_flag;
    double ref_inc_angle;
    double ref_slant_range;
    double ref_slant_range_exp;
    double rescaling_factor;
    bool bistatic_correction_applied;
    double range_sampling_rate;
    double range_bandwidth;
    double azimuth_bandwidth;
    bool multilook_flag;
    bool coregistered_stack;
    double centre_lat;
    double centre_lon;
    double centre_heading;
    double centre_heading_2;
    int first_valid_pixel;
    int last_valid_pixel;
    double slr_time_to_first_valid_pixel;
    double slr_time_to_last_valid_pixel;
    double first_valid_line_time;
    double last_valid_line_time;
    alus::cudautil::KernelArray<alus::snapengine::OrbitStateVector> orbit_state_vectors;
};

}  // namespace terraincorrection
}  // namespace alus