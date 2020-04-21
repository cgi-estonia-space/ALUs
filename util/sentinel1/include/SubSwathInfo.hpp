#pragma once
#include <iostream>
#include <fstream>
#include "CudaFriendlyObject.hpp"
#include "cuda_util.hpp"

namespace slap {

class SubSwathInfo: public CudaFriendlyObject{
private:

public:
    //subswath info
    int firstValidPixel;
    int lastValidPixel;
    double firstLineTime;
    double lastLineTime;
    double slrTimeToFirstPixel;
    double rangePixelSpacing;
    double azimuthTimeInterval;
    double radarFrequency;
    double azimuthSteeringRate;
    std::string subSwathName;

    // bursts info
    int linesPerBurst;
    int numOfBursts;
    int samplesPerBurst;
    double *burstFirstLineTime = NULL;
    double *burstLastLineTime = NULL;

    double **dopplerRate = NULL;
    double **dopplerCentroid = NULL;
    double **referenceTime =NULL;
    double **rangeDependDopplerRate = NULL;

    int dopplerSizeX, dopplerSizeY;

    double *deviceBurstFirstLineTime = NULL;
    double *deviceBurstLastLineTime = NULL;

    double *deviceDopplerRate = NULL;
    double *deviceDopplerCentroid = NULL;
    double *deviceReferenceTime =NULL;
    double *deviceRangeDependDopplerRate = NULL;

    void hostToDevice();
    void deviceToHost();
    void deviceFree();
    SubSwathInfo();
    ~SubSwathInfo();

};

}//namespace
