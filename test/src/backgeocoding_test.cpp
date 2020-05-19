#include <fstream>
#include "gmock/gmock.h"
#include "tests_common.hpp"
#include "Backgeocoding.cuh"
#include "comparators.hpp"
#include "Shapes.hpp"

using namespace slap::tests;

namespace{

class BackgeocodingTester{
private:

public:
    double *slaveTileI, *slaveTileQ;
    float *qResult;
    float *iResult;
    slap::Rectangle slaveRect;
    int tileSize;

    void readTestData(){
        int i, size;
        std::ifstream rectStream("./goods/backgeocoding/rectangle.txt");
        if(!rectStream.is_open()){
            throw std::ios::failure("rectangle.txt is not open");
        }
        rectStream >> slaveRect.x >> slaveRect.y >>slaveRect.width >> slaveRect.height;
        rectStream.close();

        std::ifstream slaveIStream("./goods/backgeocoding/slaveTileI.txt");
        std::ifstream slaveQStream("./goods/backgeocoding/slaveTileQ.txt");
        if(!slaveIStream.is_open()){
            throw std::ios::failure("slaveTileI.txt is not open.");
        }
        if(!slaveQStream.is_open()){
            throw std::ios::failure("slaveTileQ.txt is not open.");
        }

        size = slaveRect.width * slaveRect.height;

        this->slaveTileI = new double[size];
        this->slaveTileQ = new double[size];

        for(i=0; i< size; i++){
            slaveIStream >> slaveTileI[i];
            slaveQStream >> slaveTileQ[i];
        }

        slaveIStream.close();
        slaveQStream.close();

        std::ifstream qPhaseStream("./goods/backgeocoding/qPhase.txt");
        std::ifstream iPhaseStream("./goods/backgeocoding/iPhase.txt");
        if(!qPhaseStream.is_open()){
            throw std::ios::failure("qPhase.txt is not open.");
        }
        if(!iPhaseStream.is_open()){
            throw std::ios::failure("iPhase.txt is not open.");
        }
        int tileX = 100;
        int tileY = 100;
        size = tileX * tileY;
        this->tileSize = size;

        this->qResult = new float[size];
        this->iResult = new float[size];

        for(i=0; i<size; i++){
            qPhaseStream >> qResult[i];
            iPhaseStream >> iResult[i];

        }

        qPhaseStream.close();
        iPhaseStream.close();
    }

};

TEST(backgeocoding, correctness){
    slap::Backgeocoding backgeocoding;
    BackgeocodingTester tester;
    tester.readTestData();

    backgeocoding.setPlaceHolderFiles(
        "./goods/backgeocoding/params.txt",
        "./goods/backgeocoding/xPoints.txt",
        "./goods/backgeocoding/yPoints.txt"
    );
    backgeocoding.setSentinel1Placeholders(
        "./goods/backgeocoding/orbitStateVectors.txt",
        "./goods/backgeocoding/dcEstimateList.txt",
        "./goods/backgeocoding/azimuthList.txt",
        "./goods/backgeocoding/burstLineTimes.txt",
        "./goods/backgeocoding/geoLocation.txt"
    );
    backgeocoding.setSRTMPlaceholders(
        "./goods/srtm_41_01.tif",
        "./goods/srtm_42_01.tif"
    );
    backgeocoding.feedPlaceHolders();
    backgeocoding.prepareToCompute();
    backgeocoding.computeTile(tester.slaveRect, tester.slaveTileI, tester.slaveTileQ);

    const float *iResult = backgeocoding.getIResult();
    const float *qResult = backgeocoding.getQResult();

    int countI = slap::equalsArrays(iResult,tester.iResult, tester.tileSize);
    EXPECT_EQ(countI,0) << "Results I do not match. Mismatches: " <<countI << '\n';

    int countQ = slap::equalsArrays(qResult,tester.qResult, tester.tileSize);
    EXPECT_EQ(countQ,0) << "Results Q do not match. Mismatches: " <<countQ << '\n';
}


}//namespace
