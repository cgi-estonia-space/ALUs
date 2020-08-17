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

#include "gmock/gmock.h"

#include "backgeocoding.h"
#include "comparators.h"
#include "shapes.h"
#include "tests_common.hpp"

using namespace alus::tests;

namespace{

class BackgeocodingTester{
private:

public:
    double *slave_tile_i, *slave_tile_q_;
    float *q_result_;
    float *i_result_;
    alus::Rectangle slave_rect_;
    int tile_size_;

    void ReadTestData(){
        int i, size;
        std::ifstream rect_stream("./goods/backgeocoding/rectangle.txt");
        if(!rect_stream.is_open()){
            throw std::ios::failure("rectangle.txt is not open");
        }
        rect_stream >> slave_rect_.x >> slave_rect_.y >> slave_rect_.width >> slave_rect_.height;
        rect_stream.close();

        std::ifstream slave_i_stream("./goods/backgeocoding/slaveTileI.txt");
        std::ifstream slave_q_stream("./goods/backgeocoding/slaveTileQ.txt");
        if(!slave_i_stream.is_open()){
            throw std::ios::failure("slaveTileI.txt is not open.");
        }
        if(!slave_q_stream.is_open()){
            throw std::ios::failure("slaveTileQ.txt is not open.");
        }

        size = slave_rect_.width * slave_rect_.height;

        this->slave_tile_i = new double[size];
        this->slave_tile_q_ = new double[size];

        for(i=0; i< size; i++){
            slave_i_stream >> slave_tile_i[i];
            slave_q_stream >> slave_tile_q_[i];
        }

        slave_i_stream.close();
        slave_q_stream.close();

        std::ifstream q_phase_stream("./goods/backgeocoding/qPhase.txt");
        std::ifstream i_phase_stream("./goods/backgeocoding/iPhase.txt");
        if(!q_phase_stream.is_open()){
            throw std::ios::failure("qPhase.txt is not open.");
        }
        if(!i_phase_stream.is_open()){
            throw std::ios::failure("iPhase.txt is not open.");
        }
        int tile_x = 100;
        int tile_y = 100;
        size = tile_x * tile_y;
        this->tile_size_ = size;

        this->q_result_ = new float[size];
        this->i_result_ = new float[size];

        for(i=0; i<size; i++){
            q_phase_stream >> q_result_[i];
            i_phase_stream >> i_result_[i];

        }

        q_phase_stream.close();
        i_phase_stream.close();
    }

};

TEST(backgeocoding, correctness){
    alus::backgeocoding::Backgeocoding backgeocoding;
    BackgeocodingTester tester;
    tester.ReadTestData();

    backgeocoding.SetPlaceHolderFiles(
        "./goods/backgeocoding/params.txt", "./goods/backgeocoding/xPoints.txt", "./goods/backgeocoding/yPoints.txt");
    backgeocoding.SetSentinel1Placeholders(
                                           "./goods/backgeocoding/dcEstimateList.txt",
                                           "./goods/backgeocoding/azimuthList.txt",
                                           "./goods/backgeocoding/masterBurstLineTimes.txt",
                                           "./goods/backgeocoding/slaveBurstLineTimes.txt",
                                           "./goods/backgeocoding/masterGeoLocation.txt",
                                           "./goods/backgeocoding/slaveGeoLocation.txt");
    backgeocoding.SetOrbitVectorsFiles(
                     "./goods/backgeocoding/masterOrbitStateVectors.txt",
                     "./goods/backgeocoding/slaveOrbitStateVectors.txt");
    backgeocoding.SetSRTMDirectory("./goods/");
    backgeocoding.SetEGMGridFile("./goods/ww15mgh_b.grd");
    backgeocoding.FeedPlaceHolders();
    backgeocoding.PrepareToCompute();
    backgeocoding.ComputeTile(tester.slave_rect_, tester.slave_tile_i, tester.slave_tile_q_);

    const float *i_result = backgeocoding.GetIResult();
    const float *q_result = backgeocoding.GetQResult();

    int count_i = alus::EqualsArrays(i_result, tester.i_result_, tester.tile_size_);
    EXPECT_EQ(count_i,0) << "Results I do not match. Mismatches: " << count_i << '\n';

    int count_q = alus::EqualsArrays(q_result, tester.q_result_, tester.tile_size_);
    EXPECT_EQ(count_q,0) << "Results Q do not match. Mismatches: " << count_q << '\n';
}


}//namespace
