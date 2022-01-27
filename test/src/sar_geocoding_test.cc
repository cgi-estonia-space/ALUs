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

#include "../goods/S1A_IW_SLC__1SDV_20190715T160437_20190715T160504_028130_032D5B_58D6_Orb_Stack_coh_deb_orbit.h"
#include "s1tbx-commons/sar_geocoding.h"

namespace {

using alus::cuda::KernelArray;
using alus::goods::ORBIT_STATE_VECTORS;
using alus::goods::SENSOR_POSITION;
using alus::goods::SENSOR_VELOCITY;
using alus::s1tbx::sargeocoding::ComputeRangeIndexSlc;
using alus::s1tbx::sargeocoding::ComputeSlantRange;
using alus::s1tbx::sargeocoding::GetEarthPointZeroDopplerTime;
using alus::s1tbx::sargeocoding::IsDopplerTimeValid;
using alus::s1tbx::sargeocoding::IsValidCell;
using alus::snapengine::OrbitStateVectorComputation;
using alus::snapengine::PosVector;

class SarGeoCodingTest : public ::testing::Test {
public:
    // Test data copied by running terrain correction on input file
    // S1A_IW_SLC__1SDV_20190715T160437_20190715T160504_028130_032D5B_58D6_Orb_Stack_coh_deb.dim.
    // Print procedures in s1tbx repo on "snapgpu-71" branch.
    std::vector<double> const FIRST_LINE_UTC_ARGS{7135.669951395567, 7135.669951395567, 7135.669951395567,
                                                  7135.669951395567};
    std::vector<double> const LINE_TIME_INTERVAL_ARGS{2.3822903166873924E-8, 2.3822903166873924E-8,
                                                      2.3822903166873924E-8, 2.3822903166873924E-8};
    std::vector<double> const WAVELENGTH_ARGS{0.05546576, 0.05546576, 0.05546576, 0.05546576};
    std::vector<PosVector> const EARTHPOINT_ARGS{{3085549.3787867613, 1264783.8242229724, 5418767.576867359},
                                                 {3071268.9830703726, 1308500.7036828897, 5416492.407557297},
                                                 {3069968.8651965917, 1310368.109966936, 5416775.0928144},
                                                 {3092360.6043166514, 1308635.137803121, 5404519.596220633}};

    std::vector<double> const DOPPLER_TIME_RESULTS{-99999.0, 7135.669986951332, 7135.669987106099, 7135.669951395567};
};

TEST_F(SarGeoCodingTest, getEarthPointZeroDopplerTimeComputesCorrectly) {
    auto const series_length = FIRST_LINE_UTC_ARGS.size();
    const KernelArray<PosVector> sensor_positions{const_cast<PosVector*>(SENSOR_POSITION.data()),
                                                  SENSOR_POSITION.size()};
    const KernelArray<PosVector> sensor_velocity{const_cast<PosVector*>(SENSOR_VELOCITY.data()),
                                                 SENSOR_POSITION.size()};
    for (size_t i = 0; i < series_length; i++) {
        auto const result = GetEarthPointZeroDopplerTime(FIRST_LINE_UTC_ARGS.at(i), LINE_TIME_INTERVAL_ARGS.at(i),
                                                         WAVELENGTH_ARGS.at(i), EARTHPOINT_ARGS.at(i), sensor_positions,
                                                         sensor_velocity);
        EXPECT_DOUBLE_EQ(result, DOPPLER_TIME_RESULTS.at(i));
    }
}

TEST_F(SarGeoCodingTest, ComputeSlantRangeResultsAsInSnap) {
    std::vector<double> const time_args{7135.669986951332, 7135.669986692994, 7135.669986434845,
                                        7135.669986179413, 7135.669986531099, 7135.669986528665};

    const std::vector<double> slant_range_expected{836412.4827332797, 836832.2013910259, 837265.4350552851,
                                                   837688.9260249027, 841275.0188230149, 841279.0187980297};
    const std::vector<PosVector> earth_point_args{{3071268.9830703726, 1308500.7036828897, 5416492.407557297},
                                                  {3070972.528616452, 1309205.3605053576, 5416498.203012376},
                                                  {3070668.3414812526, 1309906.7204779307, 5416490.553618712},
                                                  {3070370.4826113097, 1310602.8632195268, 5416489.351410504},
                                                  {3066845.276151389, 1315745.2925090494, 5417245.928725036},
                                                  {3066842.4682139223, 1315752.0073626298, 5417246.0382013135}};
    const std::vector<PosVector> sensor_point_expected{{3658922.0283030323, 1053382.6907753784, 5954232.622894548},
                                                       {3659039.023292308, 1053468.915036083, 5954145.672627311},
                                                       {3659155.9302892475, 1053555.0759501432, 5954058.782688444},
                                                       {3659271.6043083738, 1053640.3296334823, 5953972.804162062},
                                                       {3659112.340138346, 1053522.9496627555, 5954091.181217907},
                                                       {3659113.4427363505, 1053523.7622836658, 5954090.361716833}};
    std::vector<OrbitStateVectorComputation> comp_orbits;
    comp_orbits.reserve(ORBIT_STATE_VECTORS.size());
    for (auto&& o : ORBIT_STATE_VECTORS) {
        comp_orbits.push_back({o.time_mjd_, o.x_pos_, o.y_pos_, o.z_pos_, o.x_vel_, o.y_vel_, o.z_vel_});
    }
    const KernelArray<OrbitStateVectorComputation> orbit_state_vectors{comp_orbits.data(), comp_orbits.size()};

    const auto series_size = time_args.size();
    for (size_t i = 0; i < series_size; i++) {
        PosVector sensor_point_res{};
        auto const res =
            ComputeSlantRange(time_args.at(i), orbit_state_vectors, earth_point_args.at(i), sensor_point_res);
        EXPECT_DOUBLE_EQ(res, slant_range_expected.at(i));
        EXPECT_DOUBLE_EQ(sensor_point_res.x, sensor_point_expected.at(i).x);
        EXPECT_DOUBLE_EQ(sensor_point_res.y, sensor_point_expected.at(i).y);
        EXPECT_DOUBLE_EQ(sensor_point_res.z, sensor_point_expected.at(i).z);
    }
}

TEST_F(SarGeoCodingTest, DopplerTimeValidation) {
    std::vector<double> const zero_doppler_time_args{7135.669986176958, 7135.669986951332, 7135.669986692994,
                                                     7135.669986434845, 7135.669986179413, 7135.669986528665,
                                                     7135.669986273058, 7135.669985762594};
    for (auto&& zdt : zero_doppler_time_args) {
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
    std::vector<double> const range_spacing_args{2.329562, 2.329562, 2.329562, 2.329562,
                                                 2.329562, 2.329562, 2.329562, 2.329562};
    std::vector<double> const slant_range_args{837692.989475445,  836412.4827332797, 836832.2013910259,
                                               837265.4350552851, 837688.9260249027, 841279.0187980297,
                                               841715.5319330025, 842582.278111221};
    std::vector<double> const near_edge_slant_range_args{799303.6132771898, 799303.6132771898, 799303.6132771898,
                                                         799303.6132771898, 799303.6132771898, 799303.6132771898,
                                                         799303.6132771898, 799303.6132771898};
    std::vector<double> const compute_range_index_results{16479.224935097336, 15929.54789616671,  16109.71852813367,
                                                          16295.690682667097, 16477.480637009423, 18018.582686719634,
                                                          18205.962604048633, 18578.026613600003};

    auto const series_size = compute_range_index_results.size();
    for (size_t i = 0; i < series_size; i++) {
        auto const res =
            ComputeRangeIndexSlc(range_spacing_args.at(i), slant_range_args.at(i), near_edge_slant_range_args.at(i));
        EXPECT_DOUBLE_EQ(res, compute_range_index_results.at(i));
    }
}

TEST_F(SarGeoCodingTest, IsValidCellTest) {
    int const src_max_azimuth = 1499;
    int const src_max_range = 23277;
    int const diff_lat = 0;

    double const invalid_range_index = 19514.457185293883;
    double const invalid_azimuth_index = 1499.0;
    EXPECT_FALSE(IsValidCell(invalid_range_index, invalid_azimuth_index, diff_lat, src_max_range, src_max_azimuth));

    double const valid_range_index = 19486.111233636275;
    double const valid_azimuth_index = 1498.9999618226757;

    bool valid_data_result =
        IsValidCell(valid_range_index, valid_azimuth_index, diff_lat, src_max_range, src_max_azimuth);
    EXPECT_TRUE(valid_data_result);
}

}  // namespace
