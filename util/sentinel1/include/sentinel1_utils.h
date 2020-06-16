#pragma once
#include <vector>
#include <cmath>
#include <string>
#include <fstream>

#include "shapes.h"
#include "comparators.hpp"
#include "allocators.hpp"
#include "orbit_state_vectors.h"
#include "subswath_info.h"

namespace alus {

struct AzimuthFmRate {
    double time;
    double t0;
    double c0;
    double c1;
    double c2;
};

struct DCPolynomial {
    double time;
    double t0;
    std::vector<double> dataDcPolynomial;
};

struct Sentinel1Index {
    int i0;
    int i1;
    int j0;
    int j1;
    double muX;
    double muY;
};

class Sentinel1Utils{
private:
    int numOfSubSwath;

    int isDopplerCentroidAvailable = 0;
    int isRangeDependDopplerRateAvailable = 0;
    int isOrbitAvailable = 0;
    alus::s1tbx::OrbitStateVectors *orbit{nullptr};


    std::vector<DCPolynomial> getDCEstimateList(std::string subSwathName);
    std::vector<DCPolynomial> computeDCForBurstCenters(std::vector<DCPolynomial> dcEstimateList,int subSwathIndex);
    std::vector<AzimuthFmRate> getAzimuthFmRateList(std::string subSwathName);
    DCPolynomial computeDC(double centerTime, std::vector<DCPolynomial> dcEstimateList);
    void writePlaceolderInfo(int placeholderType);
    void writeMetadataPlaceholder();
    void getProductOrbit();
    double getVelocity(double time);
    double getLatitudeValue(Sentinel1Index index, SubSwathInfo *subSwath);
    double getLongitudeValue(Sentinel1Index index, SubSwathInfo *subSwath);

    //files for placeholder data
    std::string orbitStateVectorsFile = "";
    std::string dcEstimateListFile = "";
    std::string azimuthListFile = "";
    std::string burstLineTimeFile = "";
    std::string geoLocationFile = "";

public:
    std::vector<SubSwathInfo> subSwath;
    double rangeSpacing;

    double *computeDerampDemodPhase(int subSwathIndex,int sBurstIndex,Rectangle rectangle);
    Sentinel1Index computeIndex(double azimuthTime,double slantRangeTime, SubSwathInfo *subSwath);

    void computeReferenceTime();
    void computeDopplerCentroid();
    void computeRangeDependentDopplerRate();
    void computeDopplerRate();
    double getSlantRangeTime(int x, int subSwathIndex);
    double getLatitude(double azimuthTime, double slantRangeTime, SubSwathInfo *subSwath);
    double getLongitude(double azimuthTime, double slantRangeTime, SubSwathInfo *subSwath);

    void setPlaceHolderFiles(
        std::string orbitStateVectorsFile,
        std::string dcEstimateListFile,
        std::string azimuthListFile,
        std::string burstLineTimeFile,
        std::string geoLocationFile);
    void readPlaceHolderFiles();

    Sentinel1Utils();
    Sentinel1Utils(int placeholderType);
    ~Sentinel1Utils();
};

}//namespace
