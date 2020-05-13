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

#include "sar_geocoding.h"
#include "../goods/S1A_IW_SLC__1SDV_20190715T160437_20190715T160504_028130_032D5B_58D6_Orb_Stack_coh_deb_orbit.hpp"

namespace {

using namespace slap;
using namespace slap::goods;
using namespace slap::s1tbx;
using namespace slap::s1tbx::sarGeocoding;
using namespace slap::snapEngine;

class SarGeoCodingTest : public ::testing::Test {
   public:
    // Test data copied by running terrain correction on input file
    // S1A_IW_SLC__1SDV_20190715T160437_20190715T160504_028130_032D5B_58D6_Orb_Stack_coh_deb.dim.
    // Print procedures in s1tbx repo on "snapgpu-71" branch.
    /*
    ret3 firstLineUTC:7135.669951395567 lineTimeInterval:2.3822903166873924E-8 wavelength:0.05546576
    earthPoint:3085549.3787867613,1264783.8242229724,5418767.576867359 dopplerTime:-99999.0 ret3
    firstLineUTC:7135.669951395567 lineTimeInterval:2.3822903166873924E-8 wavelength:0.05546576
    earthPoint:3085254.3565587434,1265488.1810423667,5418758.180000923 dopplerTime:-99999.0 ret3
    firstLineUTC:7135.669951395567 lineTimeInterval:2.3822903166873924E-8 wavelength:0.05546576
    earthPoint:3084960.140204468,1266192.8658246442,5418750.492520988 dopplerTime:-99999.0 ret3
    firstLineUTC:7135.669951395567 lineTimeInterval:2.3822903166873924E-8 wavelength:0.05546576
    earthPoint:3084674.8633387634,1266893.3577485026,5418754.012939339 dopplerTime:-99999.0 ret3
    firstLineUTC:7135.669951395567 lineTimeInterval:2.3822903166873924E-8 wavelength:0.05546576
    earthPoint:3086683.841047682,1265248.84660332,5417997.909457382 dopplerTime:-99999.0

    ret5 firstLineUTC:7135.669951395567 lineTimeInterval:2.3822903166873924E-8 wavelength:0.05546576
    earthPoint:3071268.9830703726,1308500.7036828897,5416492.407557297 dopplerTime:7135.669986951332 ret5
    firstLineUTC:7135.669951395567 lineTimeInterval:2.3822903166873924E-8 wavelength:0.05546576
    earthPoint:3070972.528616452,1309205.3605053576,5416498.203012376 dopplerTime:7135.669986692994 ret5
    firstLineUTC:7135.669951395567 lineTimeInterval:2.3822903166873924E-8 wavelength:0.05546576
    earthPoint:3070668.3414812526,1309906.7204779307,5416490.553618712 dopplerTime:7135.669986434845 ret5
    firstLineUTC:7135.669951395567 lineTimeInterval:2.3822903166873924E-8 wavelength:0.05546576
    earthPoint:3070370.4826113097,1310602.8632195268,5416489.351410504 dopplerTime:7135.669986179413 ret5
    firstLineUTC:7135.669951395567 lineTimeInterval:2.3822903166873924E-8 wavelength:0.05546576
    earthPoint:3066845.276151389,1315745.2925090494,5417245.928725036 dopplerTime:7135.669986531099

    ret2 firstLineUTC:7135.669951395567 lineTimeInterval:2.3822903166873924E-8 wavelength:0.05546576
    earthPoint:3069968.8651965917,1310368.109966936,5416775.0928144 dopplerTime:7135.669987106099 ret2
    firstLineUTC:7135.669951395567 lineTimeInterval:2.3822903166873924E-8 wavelength:0.05546576
    earthPoint:3069966.060764159,1310374.8279885752,5416775.187564795 dopplerTime:7135.669987106099 ret2
    firstLineUTC:7135.669951395567 lineTimeInterval:2.3822903166873924E-8 wavelength:0.05546576
    earthPoint:3069963.2563170255,1310381.546004214,5416775.282315189 dopplerTime:7135.669987106099 ret2
    firstLineUTC:7135.669951395567 lineTimeInterval:2.3822903166873924E-8 wavelength:0.05546576
    earthPoint:3069960.4518551915,1310388.2640138534,5416775.377065584 dopplerTime:7135.669987106099 ret2
    firstLineUTC:7135.669951395567 lineTimeInterval:2.3822903166873924E-8 wavelength:0.05546576
    earthPoint:3069509.7071339125,1311074.4020040652,5416858.249449753 dopplerTime:7135.669987106099

    ret1 firstLineUTC:7135.669951395567 lineTimeInterval:2.3822903166873924E-8 wavelength:0.05546576
    earthPoint:3092360.6043166514,1308635.137803121,5404519.596220633 dopplerTime:7135.669951395567 ret1
    firstLineUTC:7135.669951395567 lineTimeInterval:2.3822903166873924E-8 wavelength:0.05546576
    earthPoint:3092357.7503238223,1308641.8818912497,5404519.59622185 dopplerTime:7135.669951395567 ret1
    firstLineUTC:7135.669951395567 lineTimeInterval:2.3822903166873924E-8 wavelength:0.05546576
    earthPoint:3092354.1213876954,1308648.2980323876,5404518.232750053 dopplerTime:7135.669951395567 ret1
    firstLineUTC:7135.669951395567 lineTimeInterval:2.3822903166873924E-8 wavelength:0.05546576
    earthPoint:3092351.267365474,1308655.0421060848,5404518.232750053 dopplerTime:7135.669951395567 ret1
    firstLineUTC:7135.669951395567 lineTimeInterval:2.3822903166873924E-8 wavelength:0.05546576
    earthPoint:3092397.371973695,1308579.1303521853,5404512.594076563 dopplerTime:7135.669951395567
    */
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

TEST_F(SarGeoCodingTest, getTere){

}
}  // namespace
