#pragma once

namespace slap {

struct DeviceSubswathInfo{
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

    double *deviceBurstFirstLineTime = nullptr;
    double *deviceBurstLastLineTime = nullptr;

    double *deviceDopplerRate = nullptr;
    double *deviceDopplerCentroid = nullptr;
    double *deviceReferenceTime = nullptr;
    double *deviceRangeDependDopplerRate = nullptr;

    int dopplerSizeX, dopplerSizeY;

    // bursts info
    int linesPerBurst;
    int numOfBursts;
    int samplesPerBurst;

    // GeoLocationGridPoint
    int numOfGeoLines;
    int numOfGeoPointsPerLine;

};

}//namespace
