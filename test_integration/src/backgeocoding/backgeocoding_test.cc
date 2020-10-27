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
#include "backgeocoding_io.h"
#include "comparators.h"
#include "shapes.h"

namespace {

class BackgeocodingTester : public alus::backgeocoding::BackgeocodingIO {
   private:
   public:
    float *q_result_;
    float *i_result_;
    int tile_size_;

    void ReadTestData() {
        std::ifstream q_phase_stream("./goods/backgeocoding/qPhase.txt");
        std::ifstream i_phase_stream("./goods/backgeocoding/iPhase.txt");
        if (!q_phase_stream.is_open()) {
            throw std::ios::failure("qPhase.txt is not open.");
        }
        if (!i_phase_stream.is_open()) {
            throw std::ios::failure("iPhase.txt is not open.");
        }
        int tile_x = 100;
        int tile_y = 100;
        const size_t size = tile_x * tile_y;
        this->tile_size_ = size;

        this->q_result_ = new float[size];
        this->i_result_ = new float[size];

        for (size_t i = 0; i < size; i++) {
            q_phase_stream >> q_result_[i];
            i_phase_stream >> i_result_[i];
        }

        q_phase_stream.close();
        i_phase_stream.close();
    }
    void ReadTile(alus::Rectangle area, double *tile_i, double *tile_q) override {
        std::ifstream slave_i_stream("./goods/backgeocoding/slaveTileI.txt");
        std::ifstream slave_q_stream("./goods/backgeocoding/slaveTileQ.txt");
        if (!slave_i_stream.is_open()) {
            throw std::ios::failure("slaveTileI.txt is not open.");
        }
        if (!slave_q_stream.is_open()) {
            throw std::ios::failure("slaveTileQ.txt is not open.");
        }

        const size_t size = area.width * area.height;

        for (size_t i = 0; i < size; i++) {
            slave_i_stream >> tile_i[i];
            slave_q_stream >> tile_q[i];
        }

        slave_i_stream.close();
        slave_q_stream.close();
    }
};

TEST(backgeocoding, correctness) {
    alus::backgeocoding::Backgeocoding backgeocoding;
    BackgeocodingTester tester;
    tester.ReadTestData();

    std::vector<double> extended_amount;
    extended_amount.push_back(-0.01773467106249882);
    extended_amount.push_back(0.0);
    extended_amount.push_back(-3.770974349203243);
    extended_amount.push_back(3.8862058607542167);
    alus::Rectangle target_area = {4000, 17000, 100, 100};
    alus::Rectangle target_tile = {4000, 17000, 0, 100};

    backgeocoding.SetSentinel1Placeholders("./goods/backgeocoding/dcEstimateList.txt",
                                           "./goods/backgeocoding/azimuthList.txt",
                                           "./goods/backgeocoding/masterBurstLineTimes.txt",
                                           "./goods/backgeocoding/slaveBurstLineTimes.txt",
                                           "./goods/backgeocoding/masterGeoLocation.txt",
                                           "./goods/backgeocoding/slaveGeoLocation.txt");

    backgeocoding.SetOrbitVectorsFiles("./goods/backgeocoding/masterOrbitStateVectors.txt",
                                       "./goods/backgeocoding/slaveOrbitStateVectors.txt");
    backgeocoding.SetSRTMDirectory("./goods/");
    backgeocoding.SetEGMGridFile("./goods/backgeocoding/ww15mgh_b.grd");
    backgeocoding.FeedPlaceHolders();
    backgeocoding.PrepareToCompute();
    backgeocoding.ComputeTile(&tester, 11, 11, target_area, target_tile, extended_amount);

    const float *i_result = backgeocoding.GetIResult();
    const float *q_result = backgeocoding.GetQResult();

    size_t count_i = alus::EqualsArrays(i_result, tester.i_result_, tester.tile_size_, 0.00001);
    EXPECT_EQ(count_i, 0) << "Results I do not match. Mismatches: " << count_i << '\n';

    size_t count_q = alus::EqualsArrays(q_result, tester.q_result_, tester.tile_size_, 0.00001);
    EXPECT_EQ(count_q, 0) << "Results Q do not match. Mismatches: " << count_q << '\n';
}

}  // namespace
