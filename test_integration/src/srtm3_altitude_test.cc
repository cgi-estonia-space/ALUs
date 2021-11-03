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
#include <fstream>
#include <string>
#include <vector>

#include "gmock/gmock.h"

#include "comparators.h"
#include "cuda_friendly_object.h"
#include "cuda_util.h"
#include "shapes.h"
#include "snap-dem/dem/dataio/earth_gravitational_model96.h"
#include "srtm3_elevation_model.h"
#include "srtm3_test_util.cuh"

namespace {

using namespace alus::tests;

class Srtm3AltitudeTester : public alus::cuda::CudaFriendlyObject {
private:
    std::string test_file_name_;

public:
    std::vector<double> lats_;
    std::vector<double> lons_;
    std::vector<double> alts_;
    std::vector<double> end_results_;

    double* device_lats_{nullptr};
    double* device_lons_{nullptr};
    double* device_alts_{nullptr};

    size_t size_;

    Srtm3AltitudeTester(std::string test_file_name) { this->test_file_name_ = test_file_name; }
    ~Srtm3AltitudeTester() { this->DeviceFree(); }

    void ReadTestData() {
        std::ifstream test_data_stream(this->test_file_name_);
        if (!test_data_stream.is_open()) {
            throw std::ios::failure("srtm3 Altitude test data file not open.");
        }

        test_data_stream >> this->size_;
        this->lats_.resize(this->size_);
        this->lons_.resize(this->size_);
        this->alts_.resize(this->size_);
        this->end_results_.resize(this->size_);

        for (size_t i = 0; i < this->size_; i++) {
            test_data_stream >> this->lats_.at(i) >> this->lons_.at(i) >> this->alts_.at(i);
        }

        test_data_stream.close();
    }

    void HostToDevice() {
        CHECK_CUDA_ERR(cudaMalloc((void**)&device_lats_, this->size_ * sizeof(double)));
        CHECK_CUDA_ERR(cudaMalloc((void**)&device_lons_, this->size_ * sizeof(double)));
        CHECK_CUDA_ERR(cudaMalloc((void**)&device_alts_, this->size_ * sizeof(double)));

        CHECK_CUDA_ERR(
            cudaMemcpy(this->device_lats_, this->lats_.data(), this->size_ * sizeof(double), cudaMemcpyHostToDevice));
        CHECK_CUDA_ERR(
            cudaMemcpy(this->device_lons_, this->lons_.data(), this->size_ * sizeof(double), cudaMemcpyHostToDevice));
    }

    void DeviceToHost() {
        CHECK_CUDA_ERR(cudaMemcpy(this->end_results_.data(), this->device_alts_, this->size_ * sizeof(double),
                                  cudaMemcpyDeviceToHost));
    }

    void DeviceFree() {
        if (this->device_lats_ != nullptr) {
            cudaFree(this->device_lats_);
            this->device_lats_ = nullptr;
        }
        if (this->device_lons_ != nullptr) {
            cudaFree(this->device_lons_);
            this->device_lons_ = nullptr;
        }
        if (this->device_alts_ != nullptr) {
            cudaFree(this->device_alts_);
            this->device_alts_ = nullptr;
        }
    }
};

TEST(SRTM3, altitudeCalc) {
    Srtm3AltitudeTester tester("./goods/altitudeTestData.txt");
    tester.ReadTestData();
    tester.HostToDevice();
    dim3 block_size(512);
    dim3 grid_size(alus::cuda::GetGridDim(block_size.x, tester.size_));

    std::shared_ptr<alus::snapengine::EarthGravitationalModel96> egm_96 =
        std::make_shared<alus::snapengine::EarthGravitationalModel96>();
    egm_96->HostToDevice();

    std::vector<std::string> files{"./goods/srtm_41_01.tif", "./goods/srtm_42_01.tif"};
    alus::snapengine::Srtm3ElevationModel srtm_3_dem(files);
    srtm_3_dem.ReadSrtmTiles(egm_96);
    srtm_3_dem.HostToDevice();

    SRTM3TestData calc_data;
    calc_data.size = tester.size_;
    calc_data.tiles.array = srtm_3_dem.GetSrtmBuffersInfo();
    calc_data.tiles.size = srtm_3_dem.GetDeviceSrtm3TilesCount();

    CHECK_CUDA_ERR(LaunchSRTM3AltitudeTester(grid_size, block_size, tester.device_lats_, tester.device_lons_,
                                             tester.device_alts_, calc_data));
    tester.DeviceToHost();

    size_t count = alus::EqualsArraysd(tester.end_results_.data(), tester.alts_.data(), tester.size_, 0.00001);
    EXPECT_EQ(count, 0) << "SRTM3 altitude test results do not match. Mismatches: " << count << '\n';
}

}  // namespace
