#pragma once

#include <memory>
#include <cmath>
#include <iostream>
#include <string>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "Constants.hpp"
#include "cuda_util.hpp"
#include "sentinel1_utils.h"
#include "dataset.hpp"
#include "SRTM3ElevationModel.h"

#include "bilinear.cuh"
#include "deramp_demod.cuh"
#include "slave_pixpos.cuh"
#include "dem_formatter.cuh"



namespace alus {

class Backgeocoding{
private:
    std::vector<float> qResult;
    std::vector<float> iResult;
    std::vector<double> xPoints; //slave pixel pos x
    std::vector<double> yPoints; //slave pixel pos y
    std::vector<int> params;
    double *deviceXPoints = nullptr, *deviceYPoints = nullptr;
    double *deviceDemodI = nullptr, *deviceDemodQ = nullptr, *deviceDemodPhase = nullptr;
    float *deviceIResults = nullptr, *deviceQResults = nullptr; //I phase and Q pahse
    double *deviceSlaveI = nullptr, *deviceSlaveQ = nullptr;
    int *deviceParams = nullptr;

    int tileX, tileY, demodX, demodY, paramSize, tileSize, demodSize;

    std::unique_ptr<Sentinel1Utils> masterUtils;
    std::unique_ptr<Sentinel1Utils> slaveUtils;
    double demSamplingLat = 0.0;
    double demSamplingLon = 0.0;
    std::vector<Dataset> srtms;
    std::vector<double *> deviceSrtms;



    void allocateGPUData();
    void copySlaveTiles(double *slaveTileI, double *slaveTileQ);
    void copyGPUData();
    cudaError_t launchBilinearComp();
    cudaError_t launchDerampDemodComp(Rectangle slaveRect, int sBurstIndex);
    cudaError_t launchSlavePixPosComp(SlavePixPosData calcData);
    void getGPUEndResults();
    void prepareSrtm3Data();

    std::vector<double> computeImageGeoBoundary(SubSwathInfo *subSwath, int burstIndex,int xMin, int xMax, int yMin, int yMax);
    void computeSlavePixPos(
            int mBurstIndex,
            int sBurstIndex,
            int x0,
            int y0,
            int w,
            int h,
            std::vector<double> extendedAmount);
//            double **slavePixelPosAz,
//            double **slavePixelPosRg); add those later.



    //placeholder files
    std::string paramsFile = "../test/goods/backgeocoding/params.txt";
    std::string xPointsFile = "../test/goods/backgeocoding/xPoints.txt";
    std::string yPointsFile = "../test/goods/backgeocoding/yPoints.txt";

    std::string orbitStateVectorsFile = "../test/goods/backgeocoding/orbitStateVectors.txt";
    std::string dcEstimateListFile = "../test/goods/backgeocoding/dcEstimateList.txt";
    std::string azimuthListFile = "../test/goods/backgeocoding/azimuthList.txt";
    std::string burstLineTimeFile = "../test/goods/backgeocoding/burstLineTimes.txt";
    std::string geoLocationFile = "../test/goods/backgeocoding/geoLocation.txt";

    std::string srtm_41_01File = "../test/goods/srtm_41_01.tif";
    std::string srtm_42_01File = "../test/goods/srtm_42_01.tif";

public:

    void feedPlaceHolders();
    void prepareToCompute();
    void computeTile(Rectangle slaveRect, double *slaveTileI, double *slaveTileQ);
    Backgeocoding() = default;
    ~Backgeocoding();

    void setPlaceHolderFiles(std::string paramsFile,std::string xPointsFile, std::string yPointsFile);
    void setSentinel1Placeholders(
        std::string orbitStateVectorsFile,
        std::string dcEstimateListFile,
        std::string azimuthListFile,
        std::string burstLineTimeFile,
        std::string geoLocationFile
    );

    void setSRTMPlaceholders(std::string srtm_41_01File, std::string srtm_42_01File);

    float const* getIResult(){
        return this->iResult.data();
    }

    float const* getQResult(){
        return this->qResult.data();
    }
};

}//namespace
