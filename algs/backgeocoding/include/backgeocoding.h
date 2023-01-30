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
#include <string>
#include <string_view>
#include <vector>

#include "dem_property.h"
#include "extended_amount_computation.h"
#include "kernel_array.h"
#include "orbit_state_vector_computation.h"
#include "pointer_holders.h"
#include "s1tbx-commons/sentinel1_utils.h"

namespace alus::backgeocoding {

struct CoreComputeParams {
    Rectangle slave_rectangle;
    int s_burst_index;
    Rectangle target_area;
    size_t demod_size;
    int16_t* device_slave_i;
    int16_t* device_slave_q;

    double* device_demod_i;
    double* device_demod_q;
    double* device_demod_phase;

    double* device_x_points;
    double* device_y_points;
    // I phase and Q pahse
    float* device_i_results;
    float* device_q_results;
};

/**
 * The contents of this class originate from s1tbx module's BackGeocodingOp java class.
 * IMPORTANT: input porducts may only have 1 subswath.
 */
class Backgeocoding {
public:
    Backgeocoding() = default;
    ~Backgeocoding();
    Backgeocoding(const Backgeocoding&) = delete;  // class does not support copying(and moving)
    Backgeocoding& operator=(const Backgeocoding&) = delete;

    void PrepareToCompute(std::string_view master_metadata_file, std::string_view slave_metadata_file);
    void PrepareToCompute(std::shared_ptr<snapengine::Product> master_product,
                          std::shared_ptr<snapengine::Product> slave_product);

    void CoreCompute(const CoreComputeParams& params) const;
    Rectangle PositionCompute(int m_burst_index, int s_burst_index, Rectangle master_area, double* device_x_points,
                              double* device_y_points);

    AzimuthAndRangeBounds ComputeExtendedAmount(int x_0, int y_0, int w, int h);
    int ComputeBurstOffset();

    [[nodiscard]] int GetNrOfBursts() const { return master_utils_->subswath_.at(0)->num_of_bursts_; }
    [[nodiscard]] int GetLinesPerBurst() const { return master_utils_->subswath_.at(0)->lines_per_burst_; }
    [[nodiscard]] int GetSamplesPerBurst() const { return master_utils_->subswath_.at(0)->samples_per_burst_; }
    [[nodiscard]] int GetBurstOffset() const { return slave_burst_offset_; }
    [[nodiscard]] s1tbx::Sentinel1Utils* GetMasterUtils() const { return master_utils_.get(); }
    [[nodiscard]] s1tbx::Sentinel1Utils* GetSlaveUtils() const { return slave_utils_.get(); }

    void SetElevationData(const float* egm96_device_array, PointerArray srtm3_tiles,
                          bool mask_out_area_without_elevation, const dem::Property* dem_property) {
        egm96_device_array_ = egm96_device_array;
        srtm3_tiles_ = srtm3_tiles;
        mask_out_area_without_elevation_ = mask_out_area_without_elevation;
        dem_property_ = dem_property;
    }

private:
    /**
     * This came from chopping up the getBoundingBox function and dividing its functionality over several other
     * functions. You see we could not implement it as it was on the gpu. This struct should touch all of the parts of
     * that function.
     */
    struct CoordMinMax {
        int x_min, x_max;
        int y_min, y_max;
    };

    bool disable_reramp_ = false;  // currently not implemented
    int slave_burst_offset_;

    std::unique_ptr<s1tbx::Sentinel1Utils> master_utils_;
    std::unique_ptr<s1tbx::Sentinel1Utils> slave_utils_;
    double dem_sampling_lat_ = 0.0;
    double dem_sampling_lon_ = 0.0;
    cuda::KernelArray<snapengine::OrbitStateVectorComputation> d_master_orbit_vectors_{};
    cuda::KernelArray<snapengine::OrbitStateVectorComputation> d_slave_orbit_vectors_{};

    const float* egm96_device_array_;
    PointerArray srtm3_tiles_;
    const dem::Property* dem_property_;
    bool mask_out_area_without_elevation_ = true;

    std::vector<double> ComputeImageGeoBoundary(s1tbx::SubSwathInfo* sub_swath, int burst_index, int x_min, int x_max,
                                                int y_min, int y_max);
    bool ComputeSlavePixPos(int m_burst_index, int s_burst_index, Rectangle master_area,
                            AzimuthAndRangeBounds az_rg_bounds, CoordMinMax* coord_min_max, double* device_x_points,
                            double* device_y_points);
    void PrepareToComputeBody();
};

}  // namespace alus::backgeocoding
