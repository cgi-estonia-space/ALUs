#include <fstream>
#include <optional>
#include "comparators.h"
#include "gmock/gmock.h"
#include "sentinel1_utils.h"
#include "tests_common.hpp"

using namespace alus::tests;

namespace{

    class Sentinel1UtilsTester{
    public:
        double **doppler_rate_2_ = NULL;
        double **doppler_centroid_2_ = NULL;
        double **reference_time_2_ =NULL;
        double **range_depend_doppler_rate_2_ = NULL;

        void Read4Arrays(){
            std::ifstream doppler_rate_reader("./goods/backgeocoding/dopplerRate.txt");
            std::ifstream doppler_centroid_reader("./goods/backgeocoding/dopplerCentroid.txt");
            std::ifstream range_depend_doppler_rate_reader("./goods/backgeocoding/rangeDependDopplerRate.txt");
            std::ifstream reference_time_reader("./goods/backgeocoding/referenceTime.txt");

            int x, y, i, j;
            doppler_rate_reader >> x >> y;
            doppler_rate_2_ = alus::Allocate2DDoubleArray(x, y);

            for(i=0; i<x; i++){
                for(j=0; j<y; j++){
                    doppler_rate_reader >> doppler_rate_2_[i][j];
                }
            }

            doppler_centroid_reader >> x >> y;
            doppler_centroid_2_ = alus::Allocate2DDoubleArray(x, y);
            for(i=0; i<x; i++){
                for(j=0; j<y; j++){
                    doppler_centroid_reader >> doppler_centroid_2_[i][j];
                }
            }

            range_depend_doppler_rate_reader >> x >> y;
            range_depend_doppler_rate_2_ = alus::Allocate2DDoubleArray(x, y);
            for(i=0; i<x; i++){
                for(j=0; j<y; j++){
                    range_depend_doppler_rate_reader >> range_depend_doppler_rate_2_[i][j];
                }
            }

            reference_time_reader >> x >> y;
            reference_time_2_ = alus::Allocate2DDoubleArray(x, y);
            for(i=0; i<x; i++){
                for(j=0; j<y; j++){
                    reference_time_reader >> reference_time_2_[i][j];
                }
            }

            doppler_rate_reader.close();
            doppler_centroid_reader.close();
            range_depend_doppler_rate_reader.close();
            reference_time_reader.close();
        }

        Sentinel1UtilsTester(){

        }
        ~Sentinel1UtilsTester(){
            if(doppler_rate_2_ != NULL){
                delete[] doppler_rate_2_;
            }
            if(doppler_centroid_2_ != NULL){
                delete[] doppler_centroid_2_;
            }
            if(range_depend_doppler_rate_2_ != NULL){
                delete[] range_depend_doppler_rate_2_;
            }
            if(reference_time_2_ != NULL){
                delete[] reference_time_2_;
            }
        }
    };

    TEST(sentinel1, utils){
        alus::Sentinel1Utils utils;
        utils.SetPlaceHolderFiles("./goods/backgeocoding/orbitStateVectors.txt",
                                  "./goods/backgeocoding/dcEstimateList.txt",
                                  "./goods/backgeocoding/azimuthList.txt",
                                  "./goods/backgeocoding/burstLineTimes.txt",
                                  "./goods/backgeocoding/geoLocation.txt");
        utils.ReadPlaceHolderFiles();
        Sentinel1UtilsTester tester;
        tester.Read4Arrays();

        utils.ComputeDopplerRate();
        utils.ComputeReferenceTime();

        int count;
        std::cout << "starting comparisons." << '\n';
        ASSERT_TRUE(utils.subswath_[0].doppler_centroid_ != NULL);
        ASSERT_TRUE(tester.doppler_centroid_2_ != NULL);

        ASSERT_TRUE(utils.subswath_[0].range_depend_doppler_rate_ != NULL);
        ASSERT_TRUE(tester.range_depend_doppler_rate_2_ != NULL);

        ASSERT_TRUE(utils.subswath_[0].reference_time_ != NULL);
        ASSERT_TRUE(tester.reference_time_2_ != NULL);

        ASSERT_TRUE(utils.subswath_[0].doppler_rate_ != NULL);
        ASSERT_TRUE(tester.doppler_rate_2_ != NULL);


        count = alus::EqualsArrays2Dd(utils.subswath_[0].doppler_rate_,
                                      tester.doppler_rate_2_,
                                      utils.subswath_[0].num_of_bursts_,
                                      utils.subswath_[0].samples_per_burst_);
        EXPECT_EQ(count,0) << "Doppler Rates do not match. Mismatches: " <<count << '\n';

        count = alus::EqualsArrays2Dd(utils.subswath_[0].reference_time_,
                                      tester.reference_time_2_,
                                      utils.subswath_[0].num_of_bursts_,
                                      utils.subswath_[0].samples_per_burst_);
        EXPECT_EQ(count,0) << "Reference Times do not match. Mismatches: " <<count << '\n';

        count = alus::EqualsArrays2Dd(utils.subswath_[0].range_depend_doppler_rate_,
                                      tester.range_depend_doppler_rate_2_,
                                      utils.subswath_[0].num_of_bursts_,
                                      utils.subswath_[0].samples_per_burst_);
        EXPECT_EQ(count,0) << "Range Dependent Doppler Rates do not match. Mismatches: " <<count << '\n';

        count = alus::EqualsArrays2Dd(utils.subswath_[0].doppler_centroid_,
                                      tester.doppler_centroid_2_,
                                      utils.subswath_[0].num_of_bursts_,
                                      utils.subswath_[0].samples_per_burst_);
        EXPECT_EQ(count,0) << "Doppler Centroids do not match. Mismatches: " <<count << '\n';

    }



}//namespace
