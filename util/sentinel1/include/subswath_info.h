#pragma once
#include <iostream>
#include <fstream>
#include <vector>
#include "CudaFriendlyObject.hpp"
#include "cuda_util.hpp"

#include "subswath_info.cuh"

namespace alus {

class SubSwathInfo: public cuda::CudaFriendlyObject{
private:
    DeviceSubswathInfo devicePointersHolder;
public:
    //subswath info
    int firstValidPixel;
    int lastValidPixel;
    double firstLineTime;
    double lastLineTime;
    double slrTimeToFirstPixel;
    double slrTimeToLastPixel;
    double rangePixelSpacing;
    double azimuthTimeInterval;
    double radarFrequency;
    double azimuthSteeringRate;
    std::string subSwathName;

    // bursts info
    int linesPerBurst;
    int numOfBursts;
    int samplesPerBurst;
    double *burstFirstLineTime = nullptr; //placeholder
    double *burstLastLineTime = nullptr;  //placeholder

    double **dopplerRate = nullptr;
    double **dopplerCentroid = nullptr;
    double **referenceTime = nullptr;
    double **rangeDependDopplerRate = nullptr;

    //int dopplerSizeX, dopplerSizeY;

    // GeoLocationGridPoint
    int numOfGeoLines;
    int numOfGeoPointsPerLine;
    double **azimuthTime = nullptr; //placeholder
    double **slantRangeTime = nullptr; //placeholder
    double **latitude = nullptr;    //placeholder
    double **longitude = nullptr;   //placeholder
    double **incidenceAngle = nullptr; //placeholder

    //the packet that you can use on the gpu
    DeviceSubswathInfo *deviceSubswathInfo = nullptr;

    void hostToDevice();
    void deviceToHost();
    void deviceFree();

    SubSwathInfo();
    ~SubSwathInfo();

};

}//namespace
