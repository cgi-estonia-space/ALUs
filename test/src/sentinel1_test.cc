/**
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the Free
 * Software Foundation; either version 3 of the License, or (at your option)
 * any later version.
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for
 * more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, see http://www.gnu.org/licenses/
 */
#include <fstream>
#include <optional>

#include "comparators.h"
#include "gmock/gmock.h"
#include "sentinel1_utils.h"
#include "tests_common.hpp"
#include "allocators.h"

using namespace alus::tests;

namespace{

    class Sentinel1UtilsTester{
    public:
        double **doppler_rate_2_ = NULL;
        double **doppler_centroid_2_ = NULL;
        double **reference_time_2_ =NULL;
        double **range_depend_doppler_rate_2_ = NULL;

        void Read4Arrays(){
            std::ifstream doppler_rate_reader("./goods/backgeocoding/slaveDopplerRate.txt");
            std::ifstream doppler_centroid_reader("./goods/backgeocoding/slaveDopplerCentroid.txt");
            std::ifstream range_depend_doppler_rate_reader("./goods/backgeocoding/slaveRangeDependDopplerRate.txt");
            std::ifstream reference_time_reader("./goods/backgeocoding/slaveReferenceTime.txt");

            int x, y, i, j;
            doppler_rate_reader >> x >> y;
            doppler_rate_2_ = alus::Allocate2DArray<double>(x, y);

            for(i=0; i<x; i++){
                for(j=0; j<y; j++){
                    doppler_rate_reader >> doppler_rate_2_[i][j];
                }
            }

            doppler_centroid_reader >> x >> y;
            doppler_centroid_2_ = alus::Allocate2DArray<double>(x, y);
            for(i=0; i<x; i++){
                for(j=0; j<y; j++){
                    doppler_centroid_reader >> doppler_centroid_2_[i][j];
                }
            }

            range_depend_doppler_rate_reader >> x >> y;
            range_depend_doppler_rate_2_ = alus::Allocate2DArray<double>(x, y);
            for(i=0; i<x; i++){
                for(j=0; j<y; j++){
                    range_depend_doppler_rate_reader >> range_depend_doppler_rate_2_[i][j];
                }
            }

            reference_time_reader >> x >> y;
            reference_time_2_ = alus::Allocate2DArray<double>(x, y);
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

    /**
     * Important. Perform test with slave data as master does not have these 5 arrays.
     */
    TEST(sentinel1, utils){
        alus::s1tbx::Sentinel1Utils utils;
        utils.SetPlaceHolderFiles("./goods/backgeocoding/slaveOrbitStateVectors.txt",
                                  "./goods/backgeocoding/dcEstimateList.txt",
                                  "./goods/backgeocoding/azimuthList.txt",
                                  "./goods/backgeocoding/slaveBurstLineTimes.txt",
                                  "./goods/backgeocoding/slaveGeoLocation.txt");
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
