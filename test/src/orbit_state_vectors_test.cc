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
#include "kernel_array.h"
#include "pos_vector.h"
#include "s1tbx-commons/orbit_state_vectors.h"

namespace {

using alus::cuda::KernelArray;
using alus::goods::ORBIT_STATE_VECTORS;
using alus::snapengine::OrbitStateVectorComputation;
using alus::snapengine::PosVector;

class OrbitStateVectorsTest : public ::testing::Test {
public:
    const std::vector<double> TIME_ARGS{7135.669986176958, 7135.669986531099, 7135.669986951332,
                                        7135.669986692994, 7135.669986434845, 7135.669986179413};
    const std::vector<PosVector> POS_VECTOR_ARGS{{0.0, 0.0, 0.0},
                                                 {0.0, 0.0, 0.0},
                                                 {0.0, 0.0, 0.0},
                                                 {3658922.0283030323, 1053382.6907753784, 5954232.622894548},
                                                 {3659039.023292308, 1053468.915036083, 5954145.672627311},
                                                 {3659155.9302892475, 1053555.0759501432, 5954058.782688444}};
    const std::vector<PosVector> GET_POSITION_RESULTS{{3659272.7159376577, 1053641.1489299594, 5953971.977882909},
                                                      {3659112.340138346, 1053522.9496627555, 5954091.181217907},
                                                      {3658922.0283030323, 1053382.6907753784, 5954232.622894548},
                                                      {3659039.023292308, 1053468.915036083, 5954145.672627311},
                                                      {3659155.9302892475, 1053555.0759501432, 5954058.782688444},
                                                      {3659271.6043083738, 1053640.3296334823, 5953972.804162062}};

private:
};

TEST_F(OrbitStateVectorsTest, getPositionCalculatesCorrectly) {
    auto const series_size = POS_VECTOR_ARGS.size();
    ASSERT_EQ(series_size, GET_POSITION_RESULTS.size());
    std::vector<OrbitStateVectorComputation> comp_orbits;
    comp_orbits.reserve(ORBIT_STATE_VECTORS.size());
    for (auto&& o : ORBIT_STATE_VECTORS) {
        comp_orbits.push_back({o.time_mjd_, o.x_pos_, o.y_pos_, o.z_pos_, o.x_vel_, o.y_vel_, o.z_vel_});
    }
    const KernelArray<OrbitStateVectorComputation> orbit_state_vectors{comp_orbits.data(), comp_orbits.size()};

    for (size_t i = 0; i < series_size; i++) {
        auto const res = alus::s1tbx::orbitstatevectors::GetPosition(TIME_ARGS.at(i), orbit_state_vectors);
        EXPECT_DOUBLE_EQ(res.x, GET_POSITION_RESULTS.at(i).x);
        EXPECT_DOUBLE_EQ(res.y, GET_POSITION_RESULTS.at(i).y);
        EXPECT_DOUBLE_EQ(res.z, GET_POSITION_RESULTS.at(i).z);
    }
}

}  // namespace