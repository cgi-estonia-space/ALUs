#pragma once
#include <iostream>
#include <fstream>
#include <vector>
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

    int dopplerSizeX, dopplerSizeY;

    double *deviceBurstFirstLineTime = nullptr;
    double *deviceBurstLastLineTime = nullptr;

    double *deviceDopplerRate = nullptr;
    double *deviceDopplerCentroid = nullptr;
    double *deviceReferenceTime = nullptr;
    double *deviceRangeDependDopplerRate = nullptr;

    // GeoLocationGridPoint
    int numOfGeoLines;
    int numOfGeoPointsPerLine;
    double **azimuthTime = nullptr; //placeholder
    double **slantRangeTime = nullptr; //placeholder
    double **latitude = nullptr;    //placeholder
    double **longitude = nullptr;   //placeholder
    double **incidenceAngle = nullptr; //placeholder

    void hostToDevice();
    void deviceToHost();
    void deviceFree();

    SubSwathInfo();
    ~SubSwathInfo();

};

}//namespace
