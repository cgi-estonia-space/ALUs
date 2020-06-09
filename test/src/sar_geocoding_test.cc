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

#include <vector>

#include "gmock/gmock.h"

#include "../goods/S1A_IW_SLC__1SDV_20190715T160437_20190715T160504_028130_032D5B_58D6_Orb_Stack_coh_deb_orbit.hpp"
#include "sar_geocoding.h"

namespace {

using namespace alus;
using namespace alus::cudautil;
using namespace alus::goods;
using namespace alus::s1tbx;
using namespace alus::s1tbx::sargeocoding;
using namespace alus::snapengine;

class SarGeoCodingTest : public ::testing::Test {
   public:
    // Test data copied by running terrain correction on input file
    // S1A_IW_SLC__1SDV_20190715T160437_20190715T160504_028130_032D5B_58D6_Orb_Stack_coh_deb.dim.
    // Print procedures in s1tbx repo on "snapgpu-71" branch.
    std::vector<double> const FIRST_LINE_UTC_ARGS{
        7135.669951395567, 7135.669951395567, 7135.669951395567, 7135.669951395567};
    std::vector<double> const LINE_TIME_INTERVAL_ARGS{
        2.3822903166873924E-8, 2.3822903166873924E-8, 2.3822903166873924E-8, 2.3822903166873924E-8};
    std::vector<double> const WAVELENGTH_ARGS{0.05546576, 0.05546576, 0.05546576, 0.05546576};
    std::vector<PosVector> const EARTHPOINT_ARGS{{3085549.3787867613, 1264783.8242229724, 5418767.576867359},
                                                 {3071268.9830703726, 1308500.7036828897, 5416492.407557297},
                                                 {3069968.8651965917, 1310368.109966936, 5416775.0928144},
                                                 {3092360.6043166514, 1308635.137803121, 5404519.596220633}};

    std::vector<double> const DOPPLER_TIME_RESULTS{-99999.0, 7135.669986951332, 7135.669987106099, 7135.669951395567};
};


TEST_F(SarGeoCodingTest, getEarthPointZeroDopplerTimeComputesCorrectly) {
    auto const SERIES_LENGTH = FIRST_LINE_UTC_ARGS.size();
    const KernelArray<PosVector> sensorPositions{const_cast<PosVector*>(SENSOR_POSITION.data()),
                                                 SENSOR_POSITION.size()};
    const KernelArray<PosVector> sensorVelocity{const_cast<PosVector*>(SENSOR_VELOCITY.data()), SENSOR_POSITION.size()};
    for (size_t i = 0; i < SERIES_LENGTH; i++) {
        auto const result = GetEarthPointZeroDopplerTime(FIRST_LINE_UTC_ARGS.at(i),
                                                         LINE_TIME_INTERVAL_ARGS.at(i),
                                                         WAVELENGTH_ARGS.at(i),
                                                         EARTHPOINT_ARGS.at(i),
                                                         sensorPositions,
                                                         sensorVelocity);
        EXPECT_DOUBLE_EQ(result, DOPPLER_TIME_RESULTS.at(i));
    }
}

TEST_F(SarGeoCodingTest, ComputeSlantRangeResultsAsInSnap) {
    std::vector<double> const TIME_ARGS{7135.669986951332,
                                        7135.669986692994,
                                        7135.669986434845,
                                        7135.669986179413,
                                        7135.669986531099,
                                        7135.669986528665};
    std::vector<double> SLANT_RANGE_EXPECTED{836412.4827332797,
                                             836832.2013910259,
                                             837265.4350552851,
                                             837688.9260249027,
                                             841275.0188230149,
                                             841279.0187980297};
    std::vector<PosVector> EARTH_POINT_ARGS{{3071268.9830703726, 1308500.7036828897, 5416492.407557297},
                                            {3070972.528616452, 1309205.3605053576, 5416498.203012376},
                                            {3070668.3414812526, 1309906.7204779307, 5416490.553618712},
                                            {3070370.4826113097, 1310602.8632195268, 5416489.351410504},
                                            {3066845.276151389, 1315745.2925090494, 5417245.928725036},
                                            {3066842.4682139223, 1315752.0073626298, 5417246.0382013135}};
    std::vector<PosVector> SENSOR_POINT_EXPECTED{{3658922.0283030323, 1053382.6907753784, 5954232.622894548},
                                                 {3659039.023292308, 1053468.915036083, 5954145.672627311},
                                                 {3659155.9302892475, 1053555.0759501432, 5954058.782688444},
                                                 {3659271.6043083738, 1053640.3296334823, 5953972.804162062},
                                                 {3659112.340138346, 1053522.9496627555, 5954091.181217907},
                                                 {3659113.4427363505, 1053523.7622836658, 5954090.361716833}};
    const KernelArray<OrbitStateVector> orbitStateVectors{const_cast<OrbitStateVector*>(ORBIT_STATE_VECTORS.data()),
                                                          ORBIT_STATE_VECTORS.size()};

    const auto seriesSize = TIME_ARGS.size();
    for (size_t i = 0; i < seriesSize; i++) {
        PosVector sensorPointRes{};
        auto const res = ComputeSlantRange(TIME_ARGS.at(i), orbitStateVectors, EARTH_POINT_ARGS.at(i), sensorPointRes);
        EXPECT_DOUBLE_EQ(res, SLANT_RANGE_EXPECTED.at(i));
        EXPECT_DOUBLE_EQ(sensorPointRes.x, SENSOR_POINT_EXPECTED.at(i).x);
        EXPECT_DOUBLE_EQ(sensorPointRes.y, SENSOR_POINT_EXPECTED.at(i).y);
        EXPECT_DOUBLE_EQ(sensorPointRes.z, SENSOR_POINT_EXPECTED.at(i).z);
    }
}

TEST_F(SarGeoCodingTest, DopplerTimeValidation) {
    std::vector<double> const ZERO_DOPPLER_TIME_ARGS{7135.669986176958,
                                                     7135.669986951332,
                                                     7135.669986692994,
                                                     7135.669986434845,
                                                     7135.669986179413,
                                                     7135.669986528665,
                                                     7135.669986273058,
                                                     7135.669985762594};
    for (auto&& zdt : ZERO_DOPPLER_TIME_ARGS) {
        ASSERT_TRUE(IsDopplerTimeValid(7135.669951395567, 7135.669987106099, zdt));
    }

    // Min boundaries
    ASSERT_TRUE(IsDopplerTimeValid(7135.669951395567, 7135.669987106099, 7135.669951395567));
    ASSERT_FALSE(IsDopplerTimeValid(7135.669951395567, 7135.669987106099, 7135.669951395566));
    // Max boundaries
    ASSERT_TRUE(IsDopplerTimeValid(7135.669951395567, 7135.669987106099, 7135.669987106099));
    ASSERT_FALSE(IsDopplerTimeValid(7135.669951395567, 7135.669987106099, 7135.669987106100));
}

TEST_F(SarGeoCodingTest, ComputeRangeIndexResultsAsInSnap) {
    std::vector<double> const RANGE_SPACING_ARGS{
        2.329562, 2.329562, 2.329562, 2.329562, 2.329562, 2.329562, 2.329562, 2.329562};
    std::vector<double> const SLANT_RANGE_ARGS{837692.989475445,
                                               836412.4827332797,
                                               836832.2013910259,
                                               837265.4350552851,
                                               837688.9260249027,
                                               841279.0187980297,
                                               841715.5319330025,
                                               842582.278111221};
    std::vector<double> const NEAR_EDGE_SLANT_RANGE_ARGS{799303.6132771898,
                                                         799303.6132771898,
                                                         799303.6132771898,
                                                         799303.6132771898,
                                                         799303.6132771898,
                                                         799303.6132771898,
                                                         799303.6132771898,
                                                         799303.6132771898};
    std::vector<double> const COMPUTE_RANGE_INDEX_RESULTS{16479.224935097336,
                                                          15929.54789616671,
                                                          16109.71852813367,
                                                          16295.690682667097,
                                                          16477.480637009423,
                                                          18018.582686719634,
                                                          18205.962604048633,
                                                          18578.026613600003};

    auto const series_size = COMPUTE_RANGE_INDEX_RESULTS.size();
    for (size_t i = 0; i < series_size; i++) {
        auto const res =
            ComputeRangeIndexSlc(RANGE_SPACING_ARGS.at(i), SLANT_RANGE_ARGS.at(i), NEAR_EDGE_SLANT_RANGE_ARGS.at(i));
        EXPECT_DOUBLE_EQ(res, COMPUTE_RANGE_INDEX_RESULTS.at(i));
    }
}

}  // namespace
