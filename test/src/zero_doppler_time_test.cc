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

#include <vector>
#include <fstream>

#include "gmock/gmock.h"

#include "comparators.h"
#include "sentinel1_utils.h"
#include "pos_vector.h"
#include "cuda_util.hpp"
#include "CudaFriendlyObject.h"
#include "sar_geocoding_test.cuh"
#include "tests_common.hpp"

using namespace alus::tests;

namespace{

class ZeroDopplerTimeTester: public alus::cuda::CudaFriendlyObject{
   private:
   public:
    std::vector<alus::snapengine::PosVector> earth_points_;
    std::vector<double> line_time_intervals_;
    std::vector<double> wavelengths_;
    std::vector<double> original_zero_doppler_times_;
    std::vector<double> calcd_zero_doppler_times_;
    size_t data_size_;

    double *device_zero_doppler_times_{nullptr};
    double *device_line_time_intervals_{nullptr};
    double *device_wavelengths_{nullptr};
    alus::snapengine::PosVector *device_earth_points_{nullptr};


    ZeroDopplerTimeTester() = default;
    ~ZeroDopplerTimeTester(){
        this->DeviceFree();
    }

    void readDataFiles(){
        std::ifstream doppler_stream("./goods/backgeocoding/masterZeroDopplerTime.txt");
        if(!doppler_stream.is_open()){
            throw std::ios::failure("rectangle.txt is not open");
        }
        doppler_stream >> this->data_size_;
        this->original_zero_doppler_times_.resize(this->data_size_);
        this->calcd_zero_doppler_times_.resize(this->data_size_);
        this->line_time_intervals_.resize(this->data_size_);
        this->wavelengths_.resize(this->data_size_);
        this->earth_points_.resize(this->data_size_);

        for(size_t i=0; i<this->data_size_; i++){
            doppler_stream >> this->original_zero_doppler_times_.at(i) >> this->line_time_intervals_.at(i) >> this->wavelengths_.at(i);
            doppler_stream >> this->earth_points_.at(i).x >> this->earth_points_.at(i).y >> this->earth_points_.at(i).z;
        }

        doppler_stream.close();
    }

    void HostToDevice(){
        CHECK_CUDA_ERR(cudaMalloc((void**)&this->device_zero_doppler_times_, this->data_size_ * sizeof(double)));
        CHECK_CUDA_ERR(cudaMalloc((void**)&this->device_line_time_intervals_, this->data_size_ * sizeof(double)));
        CHECK_CUDA_ERR(cudaMalloc((void**)&this->device_wavelengths_, this->data_size_ * sizeof(double)));
        CHECK_CUDA_ERR(cudaMalloc((void**)&this->device_earth_points_, this->data_size_ * sizeof(alus::snapengine::PosVector)));

        CHECK_CUDA_ERR(cudaMemcpy(this->device_line_time_intervals_, this->line_time_intervals_.data(), this->data_size_ * sizeof(double), cudaMemcpyHostToDevice));
        CHECK_CUDA_ERR(cudaMemcpy(this->device_wavelengths_, this->wavelengths_.data(), this->data_size_ * sizeof(double), cudaMemcpyHostToDevice));
        CHECK_CUDA_ERR(cudaMemcpy(this->device_earth_points_, this->earth_points_.data(), this->data_size_ * sizeof(alus::snapengine::PosVector), cudaMemcpyHostToDevice));
    }

    void DeviceToHost() {
        CHECK_CUDA_ERR(cudaMemcpy(this->calcd_zero_doppler_times_.data(), this->device_zero_doppler_times_, this->data_size_ * sizeof(double), cudaMemcpyDeviceToHost));
    }

    void DeviceFree() {
        if(this->device_line_time_intervals_ != nullptr){
            cudaFree(this->device_line_time_intervals_);
            this->device_line_time_intervals_ = nullptr;
        }
        if(this->device_wavelengths_ != nullptr){
            cudaFree(this->device_wavelengths_);
            this->device_wavelengths_ = nullptr;
        }
        if(this->device_earth_points_ != nullptr){
            cudaFree(this->device_earth_points_);
            this->device_earth_points_ = nullptr;
        }
        if(this->device_zero_doppler_times_ != nullptr){
            cudaFree(this->device_zero_doppler_times_);
            this->device_zero_doppler_times_ = nullptr;
        }
    }
};

TEST(SarGeoCodingTestSimple, ZeroDopplerTimeTest){
    alus::s1tbx::Sentinel1Utils master_utils(1);
    master_utils.SetPlaceHolderFiles("./goods/backgeocoding/masterOrbitStateVectors.txt",
                                     "./goods/backgeocoding/dcEstimateList.txt",
                                     "./goods/backgeocoding/azimuthList.txt",
                                     "./goods/backgeocoding/masterBurstLineTimes.txt",
                                     "./goods/backgeocoding/masterGeoLocation.txt");
    master_utils.ReadPlaceHolderFiles();
    master_utils.ComputeDopplerRate();
    master_utils.ComputeReferenceTime();
    master_utils.subswath_[0].HostToDevice();
    master_utils.HostToDevice();
    alus::s1tbx::OrbitStateVectors *master_orbit = master_utils.GetOrbitStateVectors();
    master_orbit->HostToDevice();

    ZeroDopplerTimeTester tester;
    tester.readDataFiles();
    tester.HostToDevice();

    ZeroDopplerTimeData test_data;
    test_data.device_line_time_interval = tester.device_line_time_intervals_;
    test_data.device_wavelengths = tester.device_wavelengths_;
    test_data.device_earth_points = tester.device_earth_points_;
    test_data.orbit = master_orbit->device_orbit_state_vectors_;
    test_data.num_orbit_vec = master_orbit->orbitStateVectors.size();
    test_data.data_size = tester.data_size_;
    test_data.dt = master_orbit->GetDT();

    dim3 block_size(125);
    dim3 grid_size(alus::cuda::getGridDim(block_size.x,tester.data_size_));

    CHECK_CUDA_ERR(LaunchZeroDopplerTimeTest(grid_size, block_size, tester.device_zero_doppler_times_, test_data));

    tester.DeviceToHost();

    int count = alus::EqualsArraysd(tester.original_zero_doppler_times_.data(), tester.calcd_zero_doppler_times_.data(), tester.data_size_, 0.0001);
    EXPECT_EQ(count,0) << "Zero doppler time test results do not match. Mismatches: " <<count << '\n';


}

}//namespace