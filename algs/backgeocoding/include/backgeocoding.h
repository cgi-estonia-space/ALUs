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
#include <vector>

#include "backgeocoding_io.h"
#include "earth_gravitational_model96.h"
#include "extended_amount.h"
#include "kernel_array.h"
#include "orbit_state_vector_computation.h"
#include "sentinel1_utils.h"
#include "srtm3_elevation_model.h"

namespace alus {
namespace backgeocoding {

/**
 * The contents of this class originate from s1tbx module's BackGeocodingOp java class.
 */
class Backgeocoding {
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
    std::vector<float> q_result_;
    std::vector<float> i_result_;
    double *device_x_points_{nullptr}, *device_y_points_{nullptr};
    double *device_demod_i_{nullptr}, *device_demod_q_{nullptr}, *device_demod_phase_{nullptr};
    float *device_i_results_{nullptr}, *device_q_results_{nullptr};  // I phase and Q pahse
    double *device_slave_i_{nullptr}, *device_slave_q_{nullptr};

    int tile_x_, tile_y_, tile_size_, demod_size_;
    bool disable_reramp_ = 0;  // TODO: currently not implemented

    std::unique_ptr<s1tbx::Sentinel1Utils> master_utils_;
    std::unique_ptr<s1tbx::Sentinel1Utils> slave_utils_;
    double dem_sampling_lat_ = 0.0;
    double dem_sampling_lon_ = 0.0;
    std::unique_ptr<snapengine::EarthGravitationalModel96> egm96_;
    std::unique_ptr<snapengine::SRTM3ElevationModel> srtm3Dem_;

    void AllocateGPUData();
    void CopySlaveTiles(double *slave_tile_i, double *slave_tile_q);
    cudaError_t LaunchBilinearComp(Rectangle target_area,
                                   Rectangle source_area,
                                   int s_burst_index,
                                   Rectangle target_tile);
    cudaError_t LaunchDerampDemodComp(Rectangle slave_rect, int s_burst_index);
    void GetGPUEndResults();
    void PrepareSrtm3Data();

    std::vector<double> ComputeImageGeoBoundary(
        s1tbx::SubSwathInfo *sub_swath, int burst_index, int x_min, int x_max, int y_min, int y_max);
    bool ComputeSlavePixPos(int m_burst_index,
                            int s_burst_index,
                            int x0,
                            int y0,
                            int w,
                            int h,
                            std::vector<double> extended_amount,
                            CoordMinMax *coord_min_max);

    // placeholder files

    std::string slave_orbit_state_vectors_file_ = "../test/goods/backgeocoding/slaveOrbitStateVectors.txt";
    std::string master_orbit_state_vectors_file_ = "../test/goods/backgeocoding/masterOrbitStateVectors.txt";
    std::string dc_estimate_list_file_ = "../test/goods/backgeocoding/dcEstimateList.txt";
    std::string azimuth_list_file_ = "../test/goods/backgeocoding/azimuthList.txt";
    std::string master_burst_line_time_file_ = "../test/goods/backgeocoding/masterBurstLineTimes.txt";
    std::string slave_burst_line_time_file_ = "../test/goods/backgeocoding/slaveBurstLineTimes.txt";
    std::string master_geo_location_file_ = "../test/goods/backgeocoding/masterGeoLocation.txt";
    std::string slave_geo_location_file_ = "../test/goods/backgeocoding/slaveGeoLocation.txt";

    // std::string srtm_41_01File = "../test/goods/srtm_41_01.tif";
    // std::string srtm_42_01File = "../test/goods/srtm_42_01.tif";
    std::string srtms_directory_ = "../test/goods/";
    std::string grid_file_ = "../test/goods/ww15mgh_b.grd";

    cuda::KernelArray<snapengine::OrbitStateVectorComputation> d_master_orbit_vectors_{};
    cuda::KernelArray<snapengine::OrbitStateVectorComputation> d_slave_orbit_vectors_{};

   public:
    void FeedPlaceHolders();
    void PrepareToCompute();
    void ComputeTile(BackgeocodingIO *io,
                     int m_burst_index,
                     int s_burst_index,
                     Rectangle target_area,
                     Rectangle target_tile,
                     std::vector<double> extended_amount);
    Backgeocoding() = default;
    ~Backgeocoding();

    void SetSentinel1Placeholders(std::string dc_estimate_list_file,
                                  std::string azimuth_list_file,
                                  std::string master_burst_line_time_file,
                                  std::string slave_burst_line_time_file,
                                  std::string master_geo_location_file,
                                  std::string slave_geo_location_file);

    void SetOrbitVectorsFiles(std::string master_orbit_state_vectors_file, std::string slave_orbit_state_vectors_file);

    void SetSRTMDirectory(std::string directory);
    void SetEGMGridFile(std::string grid_file);

    const float *GetIResult() { return this->i_result_.data(); }

    const float *GetQResult() { return this->q_result_.data(); }

    AzimuthAndRangeBounds ComputeExtendedAmount(int x_0, int y_0, int w, int h);
};

}  // namespace backgeocoding
}  // namespace alus
