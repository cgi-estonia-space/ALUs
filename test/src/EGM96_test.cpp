#include <vector>
#include <fstream>
#include <iostream>

#include "gmock/gmock.h"
#include "tests_common.hpp"

#include "earth_gravitational_model96.h"
#include "EGM96_test.cuh"
#include "CudaFriendlyObject.hpp"
#include "cuda_util.hpp"
#include "comparators.hpp"

using namespace slap::tests;

namespace{

class EGMTester: public CudaFriendlyObject{
private:

public:
    std::vector<double> lats;
    std::vector<double> lons;
    std::vector<float> etalonResults;
    std::vector<float> endResults;
    int size;

    double *deviceLats = nullptr;
    double *deviceLons = nullptr;
    float *deviceResults = nullptr;

    EGMTester(std::string egmTestDataFilename){
        std::ifstream dataReader(egmTestDataFilename);
        dataReader >> this->size;

        this->lats.resize(size);
        this->lons.resize(size);
        this->etalonResults.resize(size);
        this->endResults.resize(size);

        for(int i=0; i<size; i++){
            dataReader >> lats[i] >> lons[i] >> etalonResults[i];
        }

        dataReader.close();
    }
    ~EGMTester(){
        this->deviceFree();
    }

    void hostToDevice(){
        CHECK_CUDA_ERR(cudaMalloc((void**)&deviceLats, this->size*sizeof(double)));
        CHECK_CUDA_ERR(cudaMalloc((void**)&deviceLons, this->size*sizeof(double)));
        CHECK_CUDA_ERR(cudaMalloc((void**)&deviceResults, this->size*sizeof(float)));

        CHECK_CUDA_ERR(cudaMemcpy(this->deviceLats, this->lats.data(), this->size*sizeof(double),cudaMemcpyHostToDevice));
        CHECK_CUDA_ERR(cudaMemcpy(this->deviceLons, this->lons.data(), this->size*sizeof(double),cudaMemcpyHostToDevice));

    }
    void deviceToHost(){
        CHECK_CUDA_ERR(cudaMemcpy(this->endResults.data(), this->deviceResults, this->size*sizeof(float), cudaMemcpyDeviceToHost));
    }
    void deviceFree(){
        cudaFree(deviceLats);
        cudaFree(deviceLons);
        cudaFree(deviceResults);
    }
};

TEST(EGM96, correctness){
    slap::snapengine::EarthGravitationalModel96 egm96("./goods/ww15mgh_b.grd");
    EGMTester tester("./goods/egm96TestData.txt");


    EXPECT_DOUBLE_EQ(13.606, egm96.egm[0][0]);
    EXPECT_DOUBLE_EQ(13.606, egm96.egm[0][1440]);

    EXPECT_DOUBLE_EQ(-29.534, egm96.egm[720][0]);
    EXPECT_DOUBLE_EQ(-29.534, egm96.egm[720][1440]);

    dim3 blockSize(20);
    dim3 gridSize(slap::getGridDim(blockSize.x,tester.size));

    tester.hostToDevice();
    egm96.hostToDevice();
    EGM96data data;
    data.MAX_LATS = slap::snapengine::earthgravitationalmodel96::MAX_LATS;
    data.MAX_LONS = slap::snapengine::earthgravitationalmodel96::MAX_LONS;
    data.size = tester.size;
    data.egm = egm96.deviceEgm;

    CHECK_CUDA_ERR(launchEGM96(gridSize, blockSize, tester.deviceLats, tester.deviceLons, tester.deviceResults, data));

    tester.deviceToHost();
    //test data file is not as accurate as I would wish
    int count = slap::equalsArrays(tester.endResults.data(), tester.etalonResults.data(), tester.size, 0.00001);
    EXPECT_EQ(count,0) << "EGM test results do not match. Mismatches: " <<count << '\n';
}


}//namespace
