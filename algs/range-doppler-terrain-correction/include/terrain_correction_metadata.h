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
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "cuda_util.h"
#include "metadata_enums.h"
#include "orbit_state_vector.h"
#include "product.h"
#include "snap-core/core/datamodel/tie_point_grid.h"
#include "spectral_band_info.h"
#include "tie_point_grid.h"

namespace alus {
namespace terraincorrection {

struct SrgrCoefficients {
    double time_mjd;
    double ground_range_origin = 0.0;
    std::vector<double> coefficient;
};

struct RangeDopplerTerrainMetadata {
    std::string product;
    metadata::ProductType product_type;
    std::string mission;
    metadata::Swath swath;
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
    metadata::Algorithm algorithm;
    double range_looks;
    double range_spacing{0.0};
    double azimuth_spacing;
    double radar_frequency;
    double line_time_interval;
    unsigned int total_size;
    double avg_scene_height;
    bool is_terrain_corrected;
    std::string dem;
    double slant_range_to_first_pixel;
    double centre_lat;
    double centre_lon;
    double centre_heading;
    int first_valid_pixel;
    int last_valid_pixel;
    double first_valid_line_time;
    double last_valid_line_time;
    std::vector<snapengine::OrbitStateVector> orbit_state_vectors2;
    double wavelength{0.0};
    std::vector<snapengine::SpectralBandInfo> band_info;
    std::vector<SrgrCoefficients> srgr_coefficients;
};

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

    Metadata(std::shared_ptr<snapengine::Product> product);

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

    static void FetchTiePoints(std::string_view tie_points_file, snapengine::tiepointgrid::TiePointGrid& tie_points,
                               std::vector<float>& buffer);
    void FillDimMetadata(std::string_view dim_metadata_file);
    void FillMetadataFrom(std::shared_ptr<snapengine::MetadataElement> master_root);
    static void FetchTiePointGrids(std::string_view dim_metadata_file,
                                   snapengine::tiepointgrid::TiePointGrid& lat_tie_point_grid,
                                   snapengine::tiepointgrid::TiePointGrid& lon_tie_point_grid);
    [[nodiscard]] static snapengine::tiepointgrid::TiePointGrid GetTiePointGrid(snapengine::TiePointGrid& grid);
    static std::vector<SrgrCoefficients> ParseSrgrCoefficients(std::shared_ptr<snapengine::MetadataElement> root);

    snapengine::tiepointgrid::TiePointGrid lat_tie_point_grid_;
    snapengine::tiepointgrid::TiePointGrid lon_tie_point_grid_;

    std::vector<float> lat_tie_points_buffer_;
    std::vector<float> lon_tie_points_buffer_;

    RangeDopplerTerrainMetadata metadata_fields_ = {};
};

}  // namespace terraincorrection
}  // namespace alus