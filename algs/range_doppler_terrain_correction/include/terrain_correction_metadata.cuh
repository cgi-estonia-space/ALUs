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

#include "terrain_correction_metadata.h"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "cuda_util.hpp"

namespace alus {
namespace terraincorrection {

/**
 * Specialisation of RangeDopplerTerrainMetadata struct for kernel.
 * All std::string values are removed and vectors are transformed into alus::cudautil::KernelArrays in order to be used with thrust library.
 */
struct RangeDopplerKernelMetadata {
    alus::metadata::ProductType product_type;
    alus::metadata::AcquisitionMode acquisition_mode;
    alus::metadata::AntennaDirection antenna_pointing;
    alus::metadata::Swath swath;
    alus::snapengine::old::Utc proc_time;
    unsigned int orbit_cycle;
    unsigned int rel_orbit;
    unsigned int abs_orbit;
    alus::snapengine::old::Utc state_vector_time;
    double incidence_near;
    double incidence_far;
    int slice_num;
    int data_take_id;
    alus::snapengine::old::Utc first_line_time;
    alus::snapengine::old::Utc last_line_time;
    double first_near_lat;
    double first_near_long;
    double first_far_lat;
    double first_far_long;
    double last_near_lat;
    double last_near_long;
    double last_far_lat;
    double last_far_long;
    alus::metadata::Pass pass;
    alus::metadata::SampleType sample_type;
    alus::metadata::Polarisation mds1_tx_rx_polar;
    alus::metadata::Polarisation mds2_tx_rx_polar;
    alus::metadata::Polarisation mds3_tx_rx_polar;
    alus::metadata::Polarisation mds4_tx_rx_polar;
    alus::metadata::Algorithm algorithm;
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

RangeDopplerKernelMetadata GetKernelMetadata(const alus::terraincorrection::RangeDopplerTerrainMetadata& cpu_metadata);

}  // namespace terraincorrection
}  // namespace alus