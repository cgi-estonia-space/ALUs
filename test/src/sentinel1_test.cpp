#include <optional>
#include <fstream>
#include "gmock/gmock.h"
#include "tests_common.hpp"
#include "sentinel1_utils.h"
#include "comparators.hpp"

using namespace alus::tests;

namespace{

    class Sentinel1UtilsTester{
    public:
        double **dopplerRate2 = NULL;
        double **dopplerCentroid2 = NULL;
        double **referenceTime2 =NULL;
        double **rangeDependDopplerRate2 = NULL;

        void read4Arrays(){
            std::ifstream dopplerRateReader("./goods/backgeocoding/dopplerRate.txt");
            std::ifstream dopplerCentroidReader("./goods/backgeocoding/dopplerCentroid.txt");
            std::ifstream rangeDependDopplerRateReader("./goods/backgeocoding/rangeDependDopplerRate.txt");
            std::ifstream referenceTimeReader("./goods/backgeocoding/referenceTime.txt");

            int x, y, i, j;
            dopplerRateReader >> x >> y;
            dopplerRate2 = alus::allocate2DDoubleArray(x,y);

            for(i=0; i<x; i++){
                for(j=0; j<y; j++){
                    dopplerRateReader >> dopplerRate2[i][j];
                }
            }

            dopplerCentroidReader >> x >> y;
            dopplerCentroid2 = alus::allocate2DDoubleArray(x,y);
            for(i=0; i<x; i++){
                for(j=0; j<y; j++){
                    dopplerCentroidReader >> dopplerCentroid2[i][j];
                }
            }

            rangeDependDopplerRateReader >> x >> y;
            rangeDependDopplerRate2 = alus::allocate2DDoubleArray(x,y);
            for(i=0; i<x; i++){
                for(j=0; j<y; j++){
                    rangeDependDopplerRateReader >> rangeDependDopplerRate2[i][j];
                }
            }

            referenceTimeReader >> x >> y;
            referenceTime2 = alus::allocate2DDoubleArray(x,y);
            for(i=0; i<x; i++){
                for(j=0; j<y; j++){
                    referenceTimeReader >> referenceTime2[i][j];
                }
            }

            dopplerRateReader.close();
            dopplerCentroidReader.close();
            rangeDependDopplerRateReader.close();
            referenceTimeReader.close();
        }

        Sentinel1UtilsTester(){

        }
        ~Sentinel1UtilsTester(){
            if(dopplerRate2 != NULL){
                delete[] dopplerRate2;
            }
            if(dopplerCentroid2 != NULL){
                delete[] dopplerCentroid2;
            }
            if(rangeDependDopplerRate2 != NULL){
                delete[] rangeDependDopplerRate2;
            }
            if(referenceTime2 != NULL){
                delete[] referenceTime2;
            }
        }
    };

    TEST(sentinel1, utils){
        alus::Sentinel1Utils utils;
        utils.setPlaceHolderFiles(
            "./goods/backgeocoding/orbitStateVectors.txt",
            "./goods/backgeocoding/dcEstimateList.txt",
            "./goods/backgeocoding/azimuthList.txt",
            "./goods/backgeocoding/burstLineTimes.txt",
            "./goods/backgeocoding/geoLocation.txt"
        );
        utils.readPlaceHolderFiles();
        Sentinel1UtilsTester tester;
        tester.read4Arrays();

        utils.computeDopplerRate();
        utils.computeReferenceTime();

        int count;
        std::cout << "starting comparisons." << '\n';
        ASSERT_TRUE(utils.subSwath[0].dopplerCentroid != NULL);
        ASSERT_TRUE(tester.dopplerCentroid2 != NULL);

        ASSERT_TRUE(utils.subSwath[0].rangeDependDopplerRate != NULL);
        ASSERT_TRUE(tester.rangeDependDopplerRate2 != NULL);

        ASSERT_TRUE(utils.subSwath[0].referenceTime != NULL);
        ASSERT_TRUE(tester.referenceTime2 != NULL);

        ASSERT_TRUE(utils.subSwath[0].dopplerRate != NULL);
        ASSERT_TRUE(tester.dopplerRate2 != NULL);


        count = alus::equalsArrays2Dd(utils.subSwath[0].dopplerRate, tester.dopplerRate2, utils.subSwath[0].numOfBursts, utils.subSwath[0].samplesPerBurst);
        EXPECT_EQ(count,0) << "Doppler Rates do not match. Mismatches: " <<count << '\n';

        count = alus::equalsArrays2Dd(utils.subSwath[0].referenceTime, tester.referenceTime2, utils.subSwath[0].numOfBursts, utils.subSwath[0].samplesPerBurst);
        EXPECT_EQ(count,0) << "Reference Times do not match. Mismatches: " <<count << '\n';

        count = alus::equalsArrays2Dd(utils.subSwath[0].rangeDependDopplerRate, tester.rangeDependDopplerRate2, utils.subSwath[0].numOfBursts, utils.subSwath[0].samplesPerBurst);
        EXPECT_EQ(count,0) << "Range Dependent Doppler Rates do not match. Mismatches: " <<count << '\n';

        count = alus::equalsArrays2Dd(utils.subSwath[0].dopplerCentroid, tester.dopplerCentroid2, utils.subSwath[0].numOfBursts, utils.subSwath[0].samplesPerBurst);
        EXPECT_EQ(count,0) << "Doppler Centroids do not match. Mismatches: " <<count << '\n';

    }



}//namespace
