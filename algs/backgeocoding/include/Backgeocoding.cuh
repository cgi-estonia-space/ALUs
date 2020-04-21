#pragma once

#include <memory>
#include <iostream>
#include <string>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>


#include "cuda_util.hpp"
#include "Sentinel1Utils.hpp"


namespace slap {

class Backgeocoding{
private:
    float *qResult = nullptr;
    float *iResult = nullptr;
    double *xPoints = nullptr; //slave pixel pos x
    double *yPoints = nullptr; //slave pixel pos y
    double *demodI = nullptr, *demodQ = nullptr, *demodPhase= nullptr;
    int *params = nullptr;
    double *deviceXPoints, *deviceYPoints, *deviceDemodI, *deviceDemodQ, *deviceDemodPhase;
    float *deviceIResults, *deviceQResults; //I phase and Q pahse
    double *deviceSlaveI, *deviceSlaveQ;
    int *deviceParams;

    int tileX, tileY, demodX, demodY, paramSize, tileSize, demodSize;

    std::unique_ptr<Sentinel1Utils> slaveUtils;


    void allocateGPUData();
    void copySlaveTiles(double *slaveTileI, double *slaveTileQ);
    void copyGPUData();
    cudaError_t launchBilinear();
    cudaError_t launchDerampDemod(Rectangle slaveRect);
    //cudaError_t launchDerampDemodPhase();
    void getGPUEndResults();

    //placeholder files
    std::string paramsFile = "../test/goods/backgeocoding/params.txt";
    std::string xPointsFile = "../test/goods/backgeocoding/xPoints.txt";
    std::string yPointsFile = "../test/goods/backgeocoding/yPoints.txt";

    std::string orbitStateVectorsFile = "../test/goods/backgeocoding/orbitStateVectors.txt";
    std::string dcEstimateListFile = "../test/goods/backgeocoding/dcEstimateList.txt";
    std::string azimuthListFile = "../test/goods/backgeocoding/azimuthList.txt";

public:

    void feedPlaceHolders();
    void prepareToCompute();
    void computeTile(Rectangle slaveRect, double *slaveTileI, double *slaveTileQ);
    Backgeocoding() = default;
    ~Backgeocoding();

    void setPlaceHolderFiles(std::string paramsFile,std::string xPointsFile, std::string yPointsFile);
    void setSentinel1Placeholders(std::string orbitStateVectorsFile, std::string dcEstimateListFile, std::string azimuthListFile);

    float *getIResult(){
        return this->iResult;
    }

    float *getQResult(){
        return this->qResult;
    }
};

}//namespace
