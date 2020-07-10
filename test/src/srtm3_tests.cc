#include <fstream>
#include <vector>
#include <string>

#include "gmock/gmock.h"

#include "CudaFriendlyObject.h"
#include "comparators.h"
#include "cuda_util.hpp"
#include "earth_gravitational_model96.h"
#include "pointer_holders.h"
#include "shapes.h"
#include "srtm3_elevation_model.h"
#include "srtm3_test_util.cuh"
#include "tests_common.hpp"

using namespace alus::tests;

namespace{

class Srtm3AltitudeTester: public alus::cuda::CudaFriendlyObject{
private:
    std::string test_file_name_;
public:
    std::vector<double> lats_;
    std::vector<double> lons_;
    std::vector<double> alts_;
    std::vector<double> end_results_;

    double *device_lats_{nullptr};
    double *device_lons_{nullptr};
    double *device_alts_{nullptr};

    size_t size_;


    Srtm3AltitudeTester(std::string test_file_name){
        this->test_file_name_ = test_file_name;
    }
    ~Srtm3AltitudeTester(){ this->DeviceFree();
    }

    void ReadTestData(){
        std::ifstream test_data_stream(this->test_file_name_);
        if(!test_data_stream.is_open()){
            throw std::ios::failure("srtm3 Altitude test data file not open.");
        }
        test_data_stream >> this->size_;
        this->lats_.resize(this->size_);
        this->lons_.resize(this->size_);
        this->alts_.resize(this->size_);
        this->end_results_.resize(this->size_);

        for(size_t i=0; i<this->size_; i++){
            test_data_stream >> this->lats_.at(i) >> this->lons_.at(i) >> this->alts_.at(i);
        }

        test_data_stream.close();
    }

    void HostToDevice(){
        CHECK_CUDA_ERR(cudaMalloc((void**)&device_lats_, this->size_*sizeof(double)));
        CHECK_CUDA_ERR(cudaMalloc((void**)&device_lons_, this->size_*sizeof(double)));
        CHECK_CUDA_ERR(cudaMalloc((void**)&device_alts_, this->size_*sizeof(double)));

        CHECK_CUDA_ERR(cudaMemcpy(this->device_lats_, this->lats_.data(), this->size_*sizeof(double),cudaMemcpyHostToDevice));
        CHECK_CUDA_ERR(cudaMemcpy(this->device_lons_, this->lons_.data(), this->size_*sizeof(double),cudaMemcpyHostToDevice));
    }

    void DeviceToHost(){
        CHECK_CUDA_ERR(cudaMemcpy(this->end_results_.data(), this->device_alts_, this->size_*sizeof(double), cudaMemcpyDeviceToHost));
    }

    void DeviceFree(){
        if(this->device_lats_ != nullptr){
            cudaFree(this->device_lats_);
            this->device_lats_ = nullptr;
        }
        if(this->device_lons_ != nullptr){
            cudaFree(this->device_lons_);
            this->device_lons_ = nullptr;
        }
        if(this->device_alts_ != nullptr){
            cudaFree(this->device_alts_);
            this->device_alts_ = nullptr;
        }
    }
};

class SRTM3TileTester{
private:
    std::string test_file_name_;
public:
    std::vector<int> xs_;
    std::vector<int> ys_;
    std::vector<double> results_;

    size_t size_;

    SRTM3TileTester(std::string test_file_name){
        this->test_file_name_ = test_file_name;
    }
    ~SRTM3TileTester(){

    }

    void ReadTestData(){
        std::ifstream test_data_reader(this->test_file_name_);
        if(!test_data_reader.is_open()){
            throw std::ios::failure("srtm3 tile test data file not open.");
        }
        test_data_reader >> this->size_;
        this->xs_.resize(this->size_);
        this->ys_.resize(this->size_);
        this->results_.resize(this->size_);

        for(size_t i=0; i<this->size_; i++){
            test_data_reader >> this->xs_.at(i) >> this->ys_.at(i) >> this->results_.at(i);
        }

        test_data_reader.close();
    }
};

TEST(SRTM3, altitudeCalc){
    Srtm3AltitudeTester tester("./goods/altitudeTestData.txt");
    tester.ReadTestData();
    tester.HostToDevice();
    dim3 block_size(512);
    dim3 grid_size(alus::cuda::getGridDim(block_size.x,tester.size_));

    alus::snapengine::EarthGravitationalModel96 egm96("./goods/ww15mgh_b.grd");
    egm96.HostToDevice();

    alus::Point srtm_41_01 = {41, 1};
    alus::Point srtm_42_01 = {42, 1};
    std::vector<alus::Point> files;
    files.push_back(srtm_41_01);
    files.push_back(srtm_42_01);
    alus::snapengine::SRTM3ElevationModel srtm3_dem(files, "./goods/");
    srtm3_dem.ReadSrtmTiles(&egm96);
    srtm3_dem.HostToDevice();

    SRTM3TestData calc_data;
    calc_data.size = tester.size_;
    calc_data.tiles.array = srtm3_dem.device_srtm3_tiles_;

    CHECK_CUDA_ERR(LaunchSRTM3AltitudeTester(
        grid_size, block_size, tester.device_lats_, tester.device_lons_, tester.device_alts_, calc_data));
    tester.DeviceToHost();

    int count = alus::EqualsArraysd(tester.end_results_.data(), tester.alts_.data(), tester.size_, 0.00001);
    EXPECT_EQ(count,0) << "SRTM3 altitude test results do not match. Mismatches: " <<count << '\n';
}

TEST(SRTM3, tileFormating){
    SRTM3TileTester tester("./goods/tileFormatTestData.txt");
    tester.ReadTestData();

    alus::snapengine::EarthGravitationalModel96 egm96("./goods/ww15mgh_b.grd");
    egm96.HostToDevice();

    alus::Point srtm_41_01 = {41, 1};
    alus::Point srtm_42_01 = {42, 1};
    std::vector<alus::Point> files;
    files.push_back(srtm_41_01);
    files.push_back(srtm_42_01);
    alus::snapengine::SRTM3ElevationModel srtm3_dem(files, "./goods/");
    srtm3_dem.ReadSrtmTiles(&egm96);
    srtm3_dem.HostToDevice();

    std::vector<double> end_tile;
    std::vector<double> end_results;
    end_results.resize(tester.size_);
    std::vector<alus::PointerHolder> tiles;
    tiles.resize(2);
    const int chosen_tile = 0;
    CHECK_CUDA_ERR(cudaMemcpy(tiles.data(), srtm3_dem.device_srtm3_tiles_, 2*sizeof(alus::PointerHolder), cudaMemcpyDeviceToHost));
    int tile_x_size = tiles.at(chosen_tile).x;
    int tile_y_size = tiles.at(chosen_tile).y;
    int tile_size = tile_x_size * tile_y_size;
    end_tile.resize(tile_size);
    CHECK_CUDA_ERR(cudaMemcpy(end_tile.data(), tiles.at(chosen_tile).pointer, tile_size *sizeof(double), cudaMemcpyDeviceToHost));

    for(size_t i=0; i<tester.size_; i++){
        end_results.at(i) = end_tile.at(tester.xs_.at(i) + tile_x_size *tester.ys_.at(i));
    }
    int count = alus::EqualsArraysd(end_results.data(), tester.results_.data(), tester.size_, 0.00001);
    EXPECT_EQ(count,0) << "SRTM3 tiling test results do not match. Mismatches: " <<count << '\n';

}

}//namespace
