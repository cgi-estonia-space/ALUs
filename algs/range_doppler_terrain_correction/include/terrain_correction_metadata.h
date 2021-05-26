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

#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

#include "cuda_util.h"
#include "metadata_enums.h"
#include "orbit_state_vector.h"
#include "snap-core/datamodel/tie_point_grid.h"
#include "spectral_band_info.h"
#include "tie_point_grid.h"

namespace alus {
namespace terraincorrection {

struct RangeDopplerTerrainMetadata {
    std::string product;
    metadata::ProductType product_type;
    std::string sph_descriptor;
    std::string mission;
    metadata::AcquisitionMode acquisition_mode;
    metadata::AntennaDirection antenna_pointing;
    std::string beams;
    metadata::Swath swath;
    std::string processing_system_identifier;
    unsigned int orbit_cycle;
    unsigned int rel_orbit;
    unsigned int abs_orbit;
    std::string vector_source;
    double incidence_near;
    double incidence_far;
    int slice_num;
    int data_take_id;
    std::shared_ptr<snapengine::Utc> first_line_time;
    std::shared_ptr<snapengine::Utc> last_line_time;
    double first_near_lat;
    double first_near_long;
    double first_far_lat;
    double first_far_long;
    double last_near_lat;
    double last_near_long;
    double last_far_lat;
    double last_far_long;
    metadata::Pass pass;
    metadata::SampleType sample_type;
    metadata::Polarisation mds1_tx_rx_polar;
    metadata::Polarisation mds2_tx_rx_polar;
    metadata::Polarisation mds3_tx_rx_polar;
    metadata::Polarisation mds4_tx_rx_polar;
    metadata::Algorithm algorithm;
    double azimuth_looks;
    double range_looks;
    double range_spacing{0.0};
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
    std::string map_projection;
    bool is_terrain_corrected;
    std::string dem;
    std::string geo_ref_system;
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
    std::string external_calibration_file;
    std::string orbit_state_vector_file;
    std::string metadata_version;
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
    std::vector<snapengine::OrbitStateVector> orbit_state_vectors;
    std::vector<snapengine::OrbitStateVector> orbit_state_vectors2;
    bool skip_bistatic_correction;
    double wavelength{0.0};
    bool is_polsar{false};
    bool near_range_on_left{true};
    double near_edge_slant_range{0.0};  // in m
    std::vector<snapengine::SpectralBandInfo> band_info;
};

// TODO: rewrite implementing SNAP internal model
class Metadata final {
public:
    struct TiePoints {
        size_t grid_width;
        size_t grid_height;
        std::vector<float> values;
    };

    Metadata() = delete;
    Metadata(std::string_view dim_metadata_file, std::string_view lat_tie_points_file,
             std::string_view lon_tie_points_file);

    [[nodiscard]] const TiePoints& GetLatTiePoints() const { return lat_tie_points_; }
    [[nodiscard]] const TiePoints& GetLonTiePoints() const { return lon_tie_points_; }
    [[nodiscard]] const RangeDopplerTerrainMetadata& GetMetadata() const { return metadata_fields_; }
    [[nodiscard]] const snapengine::tiepointgrid::TiePointGrid& GetLatTiePointGrid() const {
        return lat_tie_point_grid_;
    }
    [[nodiscard]] const snapengine::tiepointgrid::TiePointGrid& GetLonTiePointGrid() const {
        return lon_tie_point_grid_;
    }

    ~Metadata() = default;

private:
    static constexpr std::string_view LATITUDE_TIE_POINT_GRID{"latitude"};
    static constexpr std::string_view LONGITUDE_TIE_POINT_GRID{"longitude"};

    static void FetchTiePoints(std::string_view tie_points_file, TiePoints& tie_points);
    void FillDimMetadata(std::string_view dim_metadata_file);
    static void FetchTiePointGrids(std::string_view dim_metadata_file,
                                   snapengine::tiepointgrid::TiePointGrid& lat_tie_point_grid,
                                   snapengine::tiepointgrid::TiePointGrid& lon_tie_point_grid);
    [[nodiscard]] static snapengine::tiepointgrid::TiePointGrid GetTiePointGrid(const snapengine::TiePointGrid& grid);


    TiePoints lat_tie_points_;
    TiePoints lon_tie_points_;

    snapengine::tiepointgrid::TiePointGrid lat_tie_point_grid_;
    snapengine::tiepointgrid::TiePointGrid lon_tie_point_grid_;

    RangeDopplerTerrainMetadata metadata_fields_;
};

}  // namespace terraincorrection
}  // namespace alus