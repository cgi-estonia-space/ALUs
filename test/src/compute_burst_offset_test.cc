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
#include "gmock/gmock.h"

#include <fstream>

#include "../goods/compute_burst_offset_data.h"
#include "allocators.h"
#include "backgeocoding_constants.h"
#include "burst_offset_computation.h"
#include "cuda_util.h"
#include "orbit_state_vector_computation.h"
#include "shapes.h"
#include "snap-dem/dem/dataio/earth_gravitational_model96.h"
#include "srtm3_elevation_model.h"

namespace {

using alus::backgeocoding::BurstOffsetKernelArgs;
using alus::backgeocoding::INVALID_BURST_OFFSET;

using alus::snapengine::EarthGravitationalModel96;
using alus::snapengine::OrbitStateVectorComputation;
using alus::snapengine::Srtm3ElevationModel;

using alus::s1tbx::DeviceSentinel1Utils;
using alus::s1tbx::DeviceSubswathInfo;

using alus::goods::master_burst_first_line_time;
using alus::goods::master_burst_last_line_time;
using alus::goods::slave_burst_first_line_time;
using alus::goods::slave_burst_last_line_time;
using alus::goods::slave_orbit;

using alus::Allocate2DArray;
using alus::Deallocate2DArray;

class ComputeBurstOffsetTest : public ::testing::Test {
protected:
    BurstOffsetKernelArgs args_{};

private:
    std::shared_ptr<EarthGravitationalModel96> egm_96_;
    std::unique_ptr<Srtm3ElevationModel> srtm_3_dem_;
    DeviceSentinel1Utils* master_utils_{};
    DeviceSentinel1Utils* slave_utils_{};
    DeviceSubswathInfo* master_info_{};
    DeviceSubswathInfo* slave_info_{};
    double** latitude_{};
    double** longitude_{};
    size_t master_num_orbit_vec_{};
    size_t master_width_{};
    size_t master_height_{};

    OrbitStateVectorComputation* d_master_orbit_state_vector_{nullptr};
    OrbitStateVectorComputation* d_slave_orbit_state_vector_{nullptr};

    void PrepareSrtm3Data() {
        egm_96_ = std::make_shared<EarthGravitationalModel96>();
        egm_96_->HostToDevice();

        std::vector<std::string> files{"./goods/srtm_41_01.tif", "./goods/srtm_42_01.tif"};
        srtm_3_dem_ = std::make_unique<Srtm3ElevationModel>(files);
        srtm_3_dem_->ReadSrtmTiles(egm_96_);
        srtm_3_dem_->TransferToDevice();
    }
    void PrepareMasterSentinelUtils() {
        DeviceSentinel1Utils temp_utils{};
        temp_utils.line_time_interval = 2.3791160879629606e-8;  // NOLINT
        temp_utils.wavelength = 0.05546576;                     // NOLINT

        CHECK_CUDA_ERR(cudaMalloc((void**)&master_utils_, sizeof(DeviceSentinel1Utils)));
        CHECK_CUDA_ERR(cudaMemcpy(master_utils_, &temp_utils, sizeof(DeviceSentinel1Utils), cudaMemcpyHostToDevice));
    }

    void PrepareMasterSubswathInfo() {
        DeviceSubswathInfo temp_info{};

        temp_info.num_of_bursts = 19;  // NOLINT

        CHECK_CUDA_ERR(cudaMalloc((void**)&temp_info.device_burst_first_line_time,
                                  sizeof(double) * master_burst_first_line_time.size()));
        CHECK_CUDA_ERR(cudaMemcpy(temp_info.device_burst_first_line_time, master_burst_first_line_time.data(),
                                  sizeof(double) * master_burst_first_line_time.size(), cudaMemcpyHostToDevice));
        CHECK_CUDA_ERR(cudaMalloc((void**)&temp_info.device_burst_last_line_time,
                                  sizeof(double) * master_burst_last_line_time.size()));
        CHECK_CUDA_ERR(cudaMemcpy(temp_info.device_burst_last_line_time, master_burst_last_line_time.data(),
                                  sizeof(double) * master_burst_first_line_time.size(), cudaMemcpyHostToDevice));

        CHECK_CUDA_ERR(cudaMalloc((void**)&master_info_, sizeof(DeviceSubswathInfo)));
        CHECK_CUDA_ERR(cudaMemcpy(master_info_, &temp_info, sizeof(DeviceSubswathInfo), cudaMemcpyHostToDevice));
    }

    void ReadMasterGeoLocation() {
        std::ifstream geo_location_reader("./goods/backgeocoding/masterGeoLocation.txt");
        if (!geo_location_reader.is_open()) {
            throw std::ios::failure("Geo Location file not open.");
        }
        int num_of_geo_lines;
        int num_of_geo_points_per_line;

        geo_location_reader >> num_of_geo_lines >> num_of_geo_points_per_line;

        double** azimuth_time = Allocate2DArray<double>(num_of_geo_lines, num_of_geo_points_per_line);
        double** slant_range_time = Allocate2DArray<double>(num_of_geo_lines, num_of_geo_points_per_line);
        latitude_ = Allocate2DArray<double>(num_of_geo_lines, num_of_geo_points_per_line);
        longitude_ = Allocate2DArray<double>(num_of_geo_lines, num_of_geo_points_per_line);
        double** incidence_angle = Allocate2DArray<double>(num_of_geo_lines, num_of_geo_points_per_line);

        for (int i = 0; i < num_of_geo_lines; i++) {
            for (int j = 0; j < num_of_geo_points_per_line; j++) {
                geo_location_reader >> azimuth_time[i][j];
            }
        }
        for (int i = 0; i < num_of_geo_lines; i++) {
            for (int j = 0; j < num_of_geo_points_per_line; j++) {
                geo_location_reader >> slant_range_time[i][j];
            }
        }
        for (int i = 0; i < num_of_geo_lines; i++) {
            for (int j = 0; j < num_of_geo_points_per_line; j++) {
                geo_location_reader >> latitude_[i][j];
            }
        }
        for (int i = 0; i < num_of_geo_lines; i++) {
            for (int j = 0; j < num_of_geo_points_per_line; j++) {
                geo_location_reader >> longitude_[i][j];
            }
        }
        for (int i = 0; i < num_of_geo_lines; i++) {
            for (int j = 0; j < num_of_geo_points_per_line; j++) {
                geo_location_reader >> incidence_angle[i][j];
            }
        }

        geo_location_reader.close();
        master_width_ = num_of_geo_points_per_line;
        master_height_ = num_of_geo_lines;

        Deallocate2DArray(azimuth_time);
        Deallocate2DArray(slant_range_time);
        Deallocate2DArray(incidence_angle);
    }

    void ReadMasterOrbit() {
        int i;
        int count;
        OrbitStateVectorComputation temp_vector;
        std::ifstream vector_reader("./goods/backgeocoding/masterOrbitStateVectors.txt");

        if (!vector_reader.is_open()) {
            throw std::ios::failure("Vector reader is not open.");
        }
        vector_reader >> count;
        std::vector<OrbitStateVectorComputation> orbit_vectors;
        for (i = 0; i < count; i++) {
            int days{};
            int seconds{};
            int microseconds{};
            vector_reader >> days >> seconds >> microseconds;
            vector_reader >> temp_vector.timeMjd_;
            vector_reader >> temp_vector.xPos_ >> temp_vector.yPos_ >> temp_vector.zPos_;
            vector_reader >> temp_vector.xVel_ >> temp_vector.yVel_ >> temp_vector.zVel_;
            orbit_vectors.push_back(temp_vector);
        }
        vector_reader >> count;

        vector_reader.close();

        CHECK_CUDA_ERR(cudaMalloc((void**)&d_master_orbit_state_vector_, sizeof(OrbitStateVectorComputation) * count));
        CHECK_CUDA_ERR(cudaMemcpy(d_master_orbit_state_vector_, orbit_vectors.data(),
                                  sizeof(OrbitStateVectorComputation) * count, cudaMemcpyHostToDevice));
        master_num_orbit_vec_ = count;
    }

    void PrepareMasterData() {
        PrepareMasterSentinelUtils();
        PrepareMasterSubswathInfo();
        ReadMasterGeoLocation();
        ReadMasterOrbit();
    }

    void PrepareSlaveSentinelUtils() {
        DeviceSentinel1Utils temp_utils{};
        temp_utils.line_time_interval = 2.3791160879629606E-8;  // NOLINT
        temp_utils.wavelength = 0.05546576;                     // NOLINT

        CHECK_CUDA_ERR(cudaMalloc((void**)&slave_utils_, sizeof(DeviceSentinel1Utils)));
        CHECK_CUDA_ERR(cudaMemcpy(slave_utils_, &temp_utils, sizeof(DeviceSentinel1Utils), cudaMemcpyHostToDevice));
    }

    void PrepareSlaveSubswathInfo() {
        DeviceSubswathInfo temp_info{};

        temp_info.num_of_bursts = 9;  // NOLINT

        CHECK_CUDA_ERR(cudaMalloc((void**)&temp_info.device_burst_first_line_time,
                                  sizeof(double) * slave_burst_first_line_time.size()));
        CHECK_CUDA_ERR(cudaMemcpy(temp_info.device_burst_first_line_time, slave_burst_first_line_time.data(),
                                  sizeof(double) * slave_burst_first_line_time.size(), cudaMemcpyHostToDevice));
        CHECK_CUDA_ERR(cudaMalloc((void**)&temp_info.device_burst_last_line_time,
                                  sizeof(double) * slave_burst_last_line_time.size()));
        CHECK_CUDA_ERR(cudaMemcpy(temp_info.device_burst_last_line_time, slave_burst_last_line_time.data(),
                                  sizeof(double) * slave_burst_last_line_time.size(), cudaMemcpyHostToDevice));

        CHECK_CUDA_ERR(cudaMalloc((void**)&slave_info_, sizeof(DeviceSubswathInfo)));
        CHECK_CUDA_ERR(cudaMemcpy(slave_info_, &temp_info, sizeof(DeviceSubswathInfo), cudaMemcpyHostToDevice));
    }

    void PrepareSlaveData() {
        PrepareSlaveSentinelUtils();
        PrepareSlaveSubswathInfo();
        CHECK_CUDA_ERR(
            cudaMalloc((void**)&d_slave_orbit_state_vector_, sizeof(OrbitStateVectorComputation) * slave_orbit.size()));
        CHECK_CUDA_ERR(cudaMemcpy(d_slave_orbit_state_vector_, slave_orbit.data(),
                                  sizeof(OrbitStateVectorComputation) * slave_orbit.size(), cudaMemcpyHostToDevice));
    }

    void PrepareBurstOffsetKernelArgs() {
        args_.srtm3_tiles.array = srtm_3_dem_->GetBuffers();
        args_.srtm3_tiles.size = srtm_3_dem_->GetTileCount();
        args_.dem_property_ = srtm_3_dem_->GetProperties();
        args_.master_subswath_info = master_info_;
        args_.master_sentinel_utils = master_utils_;
        args_.slave_subswath_info = slave_info_;
        args_.slave_sentinel_utils = slave_utils_;
        args_.slave_orbit = d_slave_orbit_state_vector_;
        args_.master_orbit = d_master_orbit_state_vector_;
        args_.master_num_orbit_vec = static_cast<int>(master_num_orbit_vec_);
        args_.slave_num_orbit_vec = slave_orbit.size();
        args_.master_dt = 1.157407407678785E-5;  // NOLINT
        args_.slave_dt = 1.1574074073905649E-4;  // NOLINT
        args_.width = master_width_;
        args_.height = master_height_;

        size_t geo_grid_size = args_.width * args_.height;
        CHECK_CUDA_ERR(cudaMalloc(&args_.longitudes, sizeof(double) * geo_grid_size));
        CHECK_CUDA_ERR(cudaMalloc(&args_.latitudes, sizeof(double) * geo_grid_size));
        CHECK_CUDA_ERR(
            cudaMemcpy(args_.longitudes, longitude_[0], sizeof(double) * geo_grid_size, cudaMemcpyHostToDevice));
        CHECK_CUDA_ERR(
            cudaMemcpy(args_.latitudes, latitude_[0], sizeof(double) * geo_grid_size, cudaMemcpyHostToDevice));

        int burst_offset = INVALID_BURST_OFFSET;
        CHECK_CUDA_ERR(cudaMalloc(&args_.burst_offset, sizeof(int)));
        CHECK_CUDA_ERR(cudaMemcpy(args_.burst_offset, &burst_offset, sizeof(int), cudaMemcpyHostToDevice));
    }

public:
    ComputeBurstOffsetTest() {
        PrepareSrtm3Data();
        PrepareMasterData();
        PrepareSlaveData();
        PrepareBurstOffsetKernelArgs();
    }

    ~ComputeBurstOffsetTest() override {
        Deallocate2DArray(latitude_);
        Deallocate2DArray(longitude_);
        cudaFree(d_master_orbit_state_vector_);
        cudaFree(d_slave_orbit_state_vector_);
        cudaFree(master_utils_);
        cudaFree(master_info_);
        cudaFree(slave_utils_);
        cudaFree(slave_info_);
        cudaFree(args_.latitudes);
        cudaFree(args_.longitudes);
        cudaFree(args_.burst_offset);
        cudaFree(args_.master_orbit);
        cudaFree(args_.slave_orbit);
    }
};

TEST_F(ComputeBurstOffsetTest, ComputeBurstOffset) {
    constexpr int EXPECTED_BURST_OFFSET{-13};
    int burst_offset;
    cudaError_t error = LaunchBurstOffsetKernel(args_, &burst_offset);

    EXPECT_THAT(EXPECTED_BURST_OFFSET, ::testing::Eq(burst_offset));
    EXPECT_THAT(::cudaError::cudaSuccess, ::testing::Eq(error));
}
}  // namespace
