#include <fstream>
#include <vector>
#include <string>

#include "gmock/gmock.h"

#include "CudaFriendlyObject.hpp"
#include "tests_common.hpp"
#include "cuda_util.hpp"
#include "earth_gravitational_model96.h"
#include "srtm3_test_util.cuh"
#include "shapes.h"
#include "srtm3_elevation_model.h"
#include "comparators.hpp"
#include "pointer_holders.h"

using namespace alus::tests;

namespace{

class Srtm3AltitudeTester: public alus::cuda::CudaFriendlyObject{
private:
    std::string testFileName_;
public:
    std::vector<double> lats_;
    std::vector<double> lons_;
    std::vector<double> alts_;
    std::vector<double> endResults_;

    double *deviceLats_{nullptr};
    double *deviceLons_{nullptr};
    double *deviceAlts_{nullptr};

    size_t size_;


    Srtm3AltitudeTester(std::string testFileName){
        this->testFileName_ = testFileName;
    }
    ~Srtm3AltitudeTester(){
        this->deviceFree();
    }

    void ReadTestData(){
        std::ifstream testDataStream(this->testFileName_);
        if(!testDataStream.is_open()){
            throw std::ios::failure("srtm3 Altitude test data file not open.");
        }
        testDataStream >> this->size_;
        this->lats_.resize(this->size_);
        this->lons_.resize(this->size_);
        this->alts_.resize(this->size_);
        this->endResults_.resize(this->size_);

        for(size_t i=0; i<this->size_; i++){
            testDataStream >> this->lats_.at(i) >> this->lons_.at(i) >> this->alts_.at(i);
        }

        testDataStream.close();
    }

    void hostToDevice(){
        CHECK_CUDA_ERR(cudaMalloc((void**)&deviceLats_, this->size_*sizeof(double)));
        CHECK_CUDA_ERR(cudaMalloc((void**)&deviceLons_, this->size_*sizeof(double)));
        CHECK_CUDA_ERR(cudaMalloc((void**)&deviceAlts_, this->size_*sizeof(double)));

        CHECK_CUDA_ERR(cudaMemcpy(this->deviceLats_, this->lats_.data(), this->size_*sizeof(double),cudaMemcpyHostToDevice));
        CHECK_CUDA_ERR(cudaMemcpy(this->deviceLons_, this->lons_.data(), this->size_*sizeof(double),cudaMemcpyHostToDevice));
    }

    void deviceToHost(){
        CHECK_CUDA_ERR(cudaMemcpy(this->endResults_.data(), this->deviceAlts_, this->size_*sizeof(double), cudaMemcpyDeviceToHost));
    }

    void deviceFree(){
        if(this->deviceLats_ != nullptr){
            cudaFree(this->deviceLats_);
            this->deviceLats_ = nullptr;
        }
        if(this->deviceLons_ != nullptr){
            cudaFree(this->deviceLons_);
            this->deviceLons_ = nullptr;
        }
        if(this->deviceAlts_ != nullptr){
            cudaFree(this->deviceAlts_);
            this->deviceAlts_ = nullptr;
        }
    }
};

class SRTM3TileTester{
private:
    std::string testFileName_;
public:
    std::vector<int> xs_;
    std::vector<int> ys_;
    std::vector<double> results_;

    size_t size_;

    SRTM3TileTester(std::string testFileName){
        this->testFileName_ = testFileName;
    }
    ~SRTM3TileTester(){

    }

    void ReadTestData(){
        std::ifstream testDataReader(this->testFileName_);
        if(!testDataReader.is_open()){
            throw std::ios::failure("srtm3 tile test data file not open.");
        }
        testDataReader >> this->size_;
        this->xs_.resize(this->size_);
        this->ys_.resize(this->size_);
        this->results_.resize(this->size_);

        for(size_t i=0; i<this->size_; i++){
            testDataReader >> this->xs_.at(i) >> this->ys_.at(i) >> this->results_.at(i);
        }

        testDataReader.close();
    }
};

TEST(SRTM3, altitudeCalc){
    Srtm3AltitudeTester tester("./goods/altitudeTestData.txt");
    tester.ReadTestData();
    tester.hostToDevice();
    dim3 blockSize(512);
    dim3 gridSize(alus::cuda::getGridDim(blockSize.x,tester.size_));

    alus::snapengine::EarthGravitationalModel96 egm96("./goods/ww15mgh_b.grd");
    egm96.hostToDevice();

    alus::Point srtm_41_01 = {41, 1};
    alus::Point srtm_42_01 = {42, 1};
    std::vector<alus::Point> files;
    files.push_back(srtm_41_01);
    files.push_back(srtm_42_01);
    alus::snapengine::SRTM3ElevationModel srtm3Dem(files, "./goods/");
    srtm3Dem.ReadSrtmTiles(&egm96);
    srtm3Dem.hostToDevice();

    SRTM3TestData calcData;
    calcData.size = tester.size_;
    calcData.tiles.array = srtm3Dem.deviceSrtm3Tiles_;

    CHECK_CUDA_ERR(launchSRTM3AltitudeTester(gridSize, blockSize, tester.deviceLats_, tester.deviceLons_, tester.deviceAlts_, calcData));
    tester.deviceToHost();

    int count = alus::equalsArraysd(tester.endResults_.data(), tester.alts_.data(), tester.size_, 0.00001);
    EXPECT_EQ(count,0) << "SRTM3 altitude test results do not match. Mismatches: " <<count << '\n';
}

TEST(SRTM3, tileFormating){
    SRTM3TileTester tester("./goods/tileFormatTestData.txt");
    tester.ReadTestData();

    alus::snapengine::EarthGravitationalModel96 egm96("./goods/ww15mgh_b.grd");
    egm96.hostToDevice();

    alus::Point srtm_41_01 = {41, 1};
    alus::Point srtm_42_01 = {42, 1};
    std::vector<alus::Point> files;
    files.push_back(srtm_41_01);
    files.push_back(srtm_42_01);
    alus::snapengine::SRTM3ElevationModel srtm3Dem(files, "./goods/");
    srtm3Dem.ReadSrtmTiles(&egm96);
    srtm3Dem.hostToDevice();

    std::vector<double> endTile;
    std::vector<double> endResults;
    endResults.resize(tester.size_);
    std::vector<alus::PointerHolder> tiles;
    tiles.resize(2);
    const int chosenTile = 0;
    CHECK_CUDA_ERR(cudaMemcpy(tiles.data(), srtm3Dem.deviceSrtm3Tiles_, 2*sizeof(alus::PointerHolder), cudaMemcpyDeviceToHost));
    int tileXSize = tiles.at(chosenTile).x;
    int tileYSize = tiles.at(chosenTile).y;
    int tileSize = tileXSize * tileYSize;
    endTile.resize(tileSize);
    CHECK_CUDA_ERR(cudaMemcpy(endTile.data(), tiles.at(chosenTile).pointer, tileSize*sizeof(double), cudaMemcpyDeviceToHost));

    for(size_t i=0; i<tester.size_; i++){
        endResults.at(i) = endTile.at(tester.xs_.at(i) + tileXSize*tester.ys_.at(i));
    }
    int count = alus::equalsArraysd(endResults.data(), tester.results_.data(), tester.size_, 0.00001);
    EXPECT_EQ(count,0) << "SRTM3 tiling test results do not match. Mismatches: " <<count << '\n';

}

}//namespace
