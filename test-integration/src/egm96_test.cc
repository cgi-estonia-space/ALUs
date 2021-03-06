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
#include <vector>

#include "gmock/gmock.h"

#include "comparators.h"
#include "cuda_friendly_object.h"
#include "cuda_util.h"
#include "egm96_test.cuh"
#include "snap-dem/dem/dataio/earth_gravitational_model96.h"
#include "snap-dem/dem/dataio/earth_gravitational_model96_computation.h"

namespace {

class EGMTester : public alus::cuda::CudaFriendlyObject {
private:
public:
    std::vector<double> lats_;
    std::vector<double> lons_;
    std::vector<float> etalon_results_;
    std::vector<float> end_results_;
    size_t size;

    double* device_lats_{nullptr};
    double* device_lons_{nullptr};
    float* device_results_{nullptr};

    explicit EGMTester(std::string_view egm_test_data_filename) {
        std::ifstream data_reader(egm_test_data_filename.data());
        data_reader >> this->size;

        this->lats_.resize(size);
        this->lons_.resize(size);
        this->etalon_results_.resize(size);
        this->end_results_.resize(size);

        for (size_t i = 0; i < size; i++) {
            data_reader >> lats_[i] >> lons_[i] >> etalon_results_[i];
        }

        data_reader.close();
    }
    ~EGMTester() { this->DeviceFree(); }

    void HostToDevice() override {
        CHECK_CUDA_ERR(cudaMalloc((void**)&device_lats_, this->size * sizeof(double)));
        CHECK_CUDA_ERR(cudaMalloc((void**)&device_lons_, this->size * sizeof(double)));
        CHECK_CUDA_ERR(cudaMalloc((void**)&device_results_, this->size * sizeof(float)));

        CHECK_CUDA_ERR(
            cudaMemcpy(this->device_lats_, this->lats_.data(), this->size * sizeof(double), cudaMemcpyHostToDevice));
        CHECK_CUDA_ERR(
            cudaMemcpy(this->device_lons_, this->lons_.data(), this->size * sizeof(double), cudaMemcpyHostToDevice));
    }
    void DeviceToHost() override {
        CHECK_CUDA_ERR(cudaMemcpy(this->end_results_.data(), this->device_results_, this->size * sizeof(float),
                                  cudaMemcpyDeviceToHost));
    }
    void DeviceFree() override {
        cudaFree(device_lats_);
        cudaFree(device_lons_);
        cudaFree(device_results_);
    }
};

TEST(EGM96, correctness) {
    alus::snapengine::EarthGravitationalModel96 egm96{};
    EGMTester tester("./goods/egm96TestData.txt");

    auto* const host_values = egm96.GetHostValues();
    EXPECT_FLOAT_EQ(13.606, host_values[0][0]);
    EXPECT_FLOAT_EQ(13.606, host_values[0][1440]);

    EXPECT_FLOAT_EQ(-29.534, host_values[720][0]);
    EXPECT_FLOAT_EQ(-29.534, host_values[720][1440]);

    const dim3 block_size(512);
    const dim3 grid_size(alus::cuda::GetGridDim(static_cast<int>(block_size.x), static_cast<int>(tester.size)));

    tester.HostToDevice();
    egm96.HostToDevice();
    alus::tests::EGM96data data;
    data.max_lats = alus::snapengine::earthgravitationalmodel96computation::MAX_LATS;
    data.max_lons = alus::snapengine::earthgravitationalmodel96computation::MAX_LONS;
    data.size = static_cast<int>(tester.size);
    data.egm = const_cast<float*>(egm96.GetDeviceValues());

    CHECK_CUDA_ERR(
        LaunchEGM96(grid_size, block_size, tester.device_lats_, tester.device_lons_, tester.device_results_, data));

    tester.DeviceToHost();
    const float error_margin{0.0000000001};
    // test data file is not as accurate as I would wish
    size_t count = alus::EqualsArrays(tester.end_results_.data(), tester.etalon_results_.data(),
                                      static_cast<int>(tester.size), error_margin);
    EXPECT_EQ(count, 0) << "EGM test results do not match. Mismatches: " << count << '\n';
}

}  // namespace
