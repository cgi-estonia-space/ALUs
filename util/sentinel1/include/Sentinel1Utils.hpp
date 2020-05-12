#pragma once
#include <vector>
#include <cmath>
#include <string>
#include <fstream>

#include "Shapes.hpp"
#include "comparators.hpp"
#include "allocators.hpp"
#include "OrbitStateVectors.hpp"
#include "SubSwathInfo.hpp"

namespace slap {

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

class Sentinel1Utils{
private:
    int numOfSubSwath;
    int isDopplerCentroidAvailable = 0;
    int isRangeDependDopplerRateAvailable = 0;
    int isOrbitAvailable = 0;
    OrbitStateVectors *orbit = nullptr;


    std::vector<DCPolynomial> getDCEstimateList(std::string subSwathName);
    std::vector<DCPolynomial> computeDCForBurstCenters(std::vector<DCPolynomial> dcEstimateList,int subSwathIndex);
    std::vector<AzimuthFmRate> getAzimuthFmRateList(std::string subSwathName);
    DCPolynomial computeDC(double centerTime, std::vector<DCPolynomial> dcEstimateList);
    void writePlaceolderInfo();
    void writeMetadataPlaceholder();
    void getProductOrbit();
    double getVelocity(double time);

    //files for placeholder data
    std::string orbitStateVectorsFile = "../test/goods/backgeocoding/orbitStateVectors.txt";
    std::string dcEstimateListFile = "../test/goods/backgeocoding/dcEstimateList.txt";
    std::string azimuthListFile = "../test/goods/backgeocoding/azimuthList.txt";

public:
    std::vector<SubSwathInfo> subSwath;

    double *computeDerampDemodPhase(int subSwathIndex,int sBurstIndex,Rectangle rectangle);

    void computeReferenceTime();
    void computeDopplerCentroid();
    void computeRangeDependentDopplerRate();
    void computeDopplerRate();
    double getSlantRangeTime(int x, int subSwathIndex);

    void setPlaceHolderFiles(std::string orbitStateVectorsFile, std::string dcEstimateListFile, std::string azimuthListFile);

    Sentinel1Utils();
    ~Sentinel1Utils();
};

}//namespace
