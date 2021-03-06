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
#include <gmock/gmock.h>

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "../goods/S1A_IW_SLC__1SDV_20190715T160437_20190715T160504_028130_032D5B_58D6_Orb_Stack_coh_deb_orbit.h"
#include "../goods/S1A_IW_SLC__1SDV_20190715T160437_20190715T160504_028130_032D5B_58D6_Orb_Stack_coh_rectangles.h"
#include "crs_geocoding.h"
#include "cuda_util.h"
#include "gdal_management.h"
#include "get_position.h"
#include "position_data.h"
#include "snap-engine-utilities/engine-utilities/eo/constants.h"
#include "srtm3_elevation_model.h"
#include "srtm3_elevation_model_constants.h"
#include "tc_tile.h"
#include "terrain_correction.h"
#include "terrain_correction_kernel.h"
#include "terrain_correction_metadata.h"
#include "tests_common.h"
#include "tie_point_geocoding.h"

namespace alus::tests {
// Forward declare function from terrain_correction_test_kernels.cu.
void LaunchGetPositionKernel(const std::vector<double>& lat_args, const std::vector<double>& lon_args,
                             const std::vector<double>& alt_args, std::vector<s1tbx::PositionData>& sat_positions,
                             terraincorrection::GetPositionMetadata metadata,
                             const std::vector<snapengine::PosVector>& sensor_position,
                             const std::vector<snapengine::PosVector>& sensor_velocity,
                             const std::vector<snapengine::OrbitStateVectorComputation>& orbit_state_vector,
                             std::vector<bool>& results);
}  // namespace alus::tests

namespace {

using alus::Dataset;
using alus::cuda::KernelArray;
using alus::goods::ORBIT_STATE_VECTORS;
using alus::goods::SENSOR_POSITION;
using alus::goods::SENSOR_VELOCITY;
using alus::s1tbx::PositionData;
using alus::snapengine::EarthGravitationalModel96;
using alus::snapengine::OrbitStateVectorComputation;
using alus::snapengine::PosVector;
using alus::snapengine::Srtm3ElevationModel;
using alus::snapengine::geocoding::TiePointGeocoding;
using alus::snapengine::tiepointgrid::TiePointGrid;
using alus::terraincorrection::CalculateOrbitStateVectorLUT;
using alus::terraincorrection::GetPositionMetadata;
using alus::terraincorrection::Metadata;
using alus::terraincorrection::RangeDopplerTerrainMetadata;
using alus::terraincorrection::TerrainCorrection;
using alus::tests::COH_1_TIF;
using alus::tests::DEM_PATH_1;
using alus::tests::LaunchGetPositionKernel;
using alus::tests::TC_OUTPUT;

class TerrainCorrectionTest : public ::testing::Test {
public:
    TerrainCorrectionTest() {
        coh_ds_ = std::make_optional<Dataset<double>>(COH_1_TIF);
        coh_ds_.value().LoadRasterBand(1);
        dem_ds_ = std::make_optional<Dataset<double>>(DEM_PATH_1);
        dem_ds_.value().LoadRasterBand(1);
        metadata_ = std::make_optional<Metadata>(main_metadata_file_, lat_tie_points_file_, lon_tie_points_file_);
        tc_metadata_ = metadata_->GetMetadata();

        egm_96_ = std::make_shared<EarthGravitationalModel96>();
        egm_96_->HostToDevice();

        std::vector<std::string> files{"./goods/srtm_41_01.tif", "./goods/srtm_42_01.tif"};
        srtm_3_model_ = std::make_unique<Srtm3ElevationModel>(files);
        srtm_3_model_->ReadSrtmTiles(egm_96_);
        srtm_3_model_->HostToDevice();
    }

    std::optional<Dataset<double>> coh_ds_;
    std::optional<Dataset<double>> coh_data_ds_;
    std::optional<Dataset<double>> dem_ds_;
    std::optional<Metadata> metadata_;
    std::optional<RangeDopplerTerrainMetadata> tc_metadata_;
    std::unique_ptr<Srtm3ElevationModel> srtm_3_model_;
    std::shared_ptr<EarthGravitationalModel96> egm_96_;
    const size_t srtm_3_tiles_length_{2};

protected:
private:
    const std::string main_metadata_file_{
        "goods/S1A_IW_SLC__1SDV_20190715T160437_20190715T160504_028130_032D5B_58D6_Orb_Stack_coh_deb.dim"};
    const std::string lat_tie_points_file_{
        "goods/S1A_IW_SLC__1SDV_20190715T160437_20190715T160504_028130_032D5B_58D6_Orb_Stack_coh_deb"
        ".data/tie_point_grids/latitude.img"};
    const std::string lon_tie_points_file_{
        "goods/S1A_IW_SLC__1SDV_20190715T160437_20190715T160504_028130_032D5B_58D6_Orb_Stack_coh_deb"
        ".data/tie_point_grids/longitude.img"};
};

TEST_F(TerrainCorrectionTest, getPositionTrueScenario) {
    std::vector<double> const lats_true{58.52938269941166, 58.52938269941166, 58.52938269941166, 58.52938269941166,
                                        58.52938269941166, 58.52938269941166, 58.52938269941166, 58.52938269941166,
                                        58.52938269941166, 58.52925774375564};
    std::vector<double> const lons_true{23.114453402942733, 23.114578358598756, 23.114703314254776, 23.114828269910795,
                                        23.11495322556682,  23.115078181222838, 23.11520313687886,  23.11532809253488,
                                        23.1154530481909,   23.113203846382525};
    std::vector<double> const alts_true{26.655392062820304, 26.766483052830047, 26.877574042861347, 26.98866503287109,
                                        27.099756022880833, 27.16567189351999,  27.204560294821366, 27.243448696122737,
                                        27.28233709742411,  25.688039075802656};
    std::vector<PositionData> const pos_data_true{{{3069968.8651965917, 1310368.109966936, 5416775.0928144},
                                                   {3658851.937123117, 1053331.0349233549, 5954284.711984713},
                                                   1499.0,
                                                   16489.057561014797,
                                                   837715.8951871425},
                                                  {{3069966.060764159, 1310374.8279885752, 5416775.187564795},
                                                   {3658851.937123117, 1053331.0349233549, 5954284.711984713},
                                                   1499.0,
                                                   16490.76257450734,
                                                   837719.8671217842},
                                                  {{3069963.2563170255, 1310381.546004214, 5416775.282315189},
                                                   {3658851.937123117, 1053331.0349233549, 5954284.711984713},
                                                   1499.0,
                                                   16492.467610722975,
                                                   837723.8391093608},
                                                  {{3069960.4518551915, 1310388.2640138534, 5416775.377065584},
                                                   {3658851.937123117, 1053331.0349233549, 5954284.711984713},
                                                   1499.0,
                                                   16494.172669660762,
                                                   837727.81114987},
                                                  {{3069957.6473786565, 1310394.9820174929, 5416775.471815978},
                                                   {3658851.937123117, 1053331.0349233549, 5954284.711984713},
                                                   1499.0,
                                                   16495.87775132104,
                                                   837731.7832433127},
                                                  {{3069954.8211966213, 1310401.6907564746, 5416775.528036151},
                                                   {3658851.9375350126, 1053331.0352269127, 5954284.711678611},
                                                   1498.9999618226757,
                                                   16497.598793672605,
                                                   837735.7925181753},
                                                  {{3069951.9820227185, 1310408.3939500444, 5416775.56120438},
                                                   {3658851.9375350126, 1053331.0352269127, 5954284.711678611},
                                                   1498.9999618226757,
                                                   16499.329393917273,
                                                   837739.8240587425},
                                                  {{3069949.1428341805, 1310415.0971374626, 5416775.594372609},
                                                   {3658851.937946909, 1053331.0355304708, 5954284.7113725105},
                                                   1498.9999236453511,
                                                   16501.060016592295,
                                                   837743.8556515626},
                                                  {{3069946.303631006, 1310421.8003187296, 5416775.627540837},
                                                   {3658851.937946909, 1053331.0355304708, 5954284.7113725105},
                                                   1498.9999236453511,
                                                   16502.790661701973,
                                                   837747.8872966456},
                                                  {{3070007.895933579, 1310305.6188605186, 5416767.001437339},
                                                   {3658851.937123117, 1053331.0349233549, 5954284.711984713},
                                                   1499.0,
                                                   16471.278523574896,
                                                   837674.477817126}};

    std::vector<OrbitStateVectorComputation> comp_orbits;
    comp_orbits.reserve(ORBIT_STATE_VECTORS.size());
    for (auto&& o : ORBIT_STATE_VECTORS) {
        comp_orbits.push_back({o.time_mjd_, o.x_pos_, o.y_pos_, o.z_pos_, o.x_vel_, o.y_vel_, o.z_vel_});
    }

    const KernelArray<OrbitStateVectorComputation> orbit_state_vectors{
        const_cast<OrbitStateVectorComputation*>(comp_orbits.data()), comp_orbits.size()};
    const KernelArray<PosVector> sensor_positions{const_cast<PosVector*>(SENSOR_POSITION.data()),
                                                  SENSOR_POSITION.size()};
    const KernelArray<PosVector> sensor_velocity{const_cast<PosVector*>(SENSOR_VELOCITY.data()),
                                                 SENSOR_VELOCITY.size()};

    std::vector<double> osv_lookup = CalculateOrbitStateVectorLUT(comp_orbits);

    KernelArray<double> osv_lut = {osv_lookup.data(), osv_lookup.size()};

    const GetPositionMetadata metadata{7135.669951395567, 2.3822903166873924E-8, 0.05546576,
                                       2.329562,          799303.6132771898,     sensor_positions,
                                       sensor_velocity,   orbit_state_vectors,   osv_lut};
    const auto series_size = pos_data_true.size();
    for (size_t i = 0; i < series_size; i++) {
        PositionData pos_data{};
        const auto ret = GetPosition(lats_true.at(i), lons_true.at(i), alts_true.at(i), pos_data, metadata);
        EXPECT_TRUE(ret);
        EXPECT_DOUBLE_EQ(pos_data.earth_point.x, pos_data_true.at(i).earth_point.x);
        EXPECT_DOUBLE_EQ(pos_data.earth_point.y, pos_data_true.at(i).earth_point.y);
        EXPECT_DOUBLE_EQ(pos_data.earth_point.z, pos_data_true.at(i).earth_point.z);
        EXPECT_DOUBLE_EQ(pos_data.sensor_pos.x, pos_data_true.at(i).sensor_pos.x);
        EXPECT_DOUBLE_EQ(pos_data.sensor_pos.y, pos_data_true.at(i).sensor_pos.y);
        EXPECT_DOUBLE_EQ(pos_data.sensor_pos.z, pos_data_true.at(i).sensor_pos.z);
        EXPECT_DOUBLE_EQ(pos_data.azimuth_index, pos_data_true.at(i).azimuth_index);
        EXPECT_DOUBLE_EQ(pos_data.range_index, pos_data_true.at(i).range_index);
        EXPECT_DOUBLE_EQ(pos_data.slant_range, pos_data_true.at(i).slant_range);
    }
}

TEST_F(TerrainCorrectionTest, getPositionFalseScenario) {
    std::vector<double> const lats_false{58.57074302155462, 58.57074302155462, 58.57074302155462, 58.57074302155462,
                                         58.57074302155462, 58.57074302155462, 58.57074302155462, 58.57074302155462,
                                         58.57074302155462, 58.57074302155462};
    std::vector<double> const lons_false{22.29012094017203,  22.29024589582805,  22.29037085148407,  22.290495807140093,
                                         22.290620762796113, 22.290745718452133, 22.290870674108156, 22.290995629764176,
                                         22.2911205854202,   22.29124554107622};
    std::vector<double> const alts_false{32.416427961115495, 32.38403161384701, 32.3516352665848,   32.319238919316305,
                                         32.286842572054105, 32.2544462247919,  32.182097076725086, 32.01600429552241,
                                         31.849911514287488, 31.683818733084806};
    std::vector<PositionData> const pos_data_false{
        {{3084869.388681489, 1264575.8165823026, 5419183.743972956}, {0.0, 0.0, 0.0}, 0.0, 0.0, 0.0},
        {{3084866.615145824, 1264582.5379252795, 5419183.716329651}, {0.0, 0.0, 0.0}, 0.0, 0.0, 0.0},
        {{3084863.8415955133, 1264589.2592621732, 5419183.688686345}, {0.0, 0.0, 0.0}, 0.0, 0.0, 0.0},
        {{3084861.0680305585, 1264595.9805929847, 5419183.6610430395}, {0.0, 0.0, 0.0}, 0.0, 0.0, 0.0},
        {{3084858.294450959, 1264602.7019177126, 5419183.633399734}, {0.0, 0.0, 0.0}, 0.0, 0.0, 0.0},
        {{3084855.5208567153, 1264609.423236358, 5419183.605756428}, {0.0, 0.0, 0.0}, 0.0, 0.0, 0.0},
        {{3084852.727971484, 1264616.1366467036, 5419183.544022011}, {0.0, 0.0, 0.0}, 0.0, 0.0, 0.0},
        {{3084849.889842454, 1264622.8315093429, 5419183.402297591}, {0.0, 0.0, 0.0}, 0.0, 0.0, 0.0},
        {{3084847.0516988942, 1264629.526365618, 5419183.260573173}, {0.0, 0.0, 0.0}, 0.0, 0.0, 0.0},
        {{3084844.2135408055, 1264636.2212155282, 5419183.118848753}, {0.0, 0.0, 0.0}, 0.0, 0.0, 0.0}};

    std::vector<OrbitStateVectorComputation> comp_orbits;
    comp_orbits.reserve(ORBIT_STATE_VECTORS.size());
    for (auto&& o : ORBIT_STATE_VECTORS) {
        comp_orbits.push_back({o.time_mjd_, o.x_pos_, o.y_pos_, o.z_pos_, o.x_vel_, o.y_vel_, o.z_vel_});
    }

    const KernelArray<OrbitStateVectorComputation> orbit_state_vectors{
        const_cast<OrbitStateVectorComputation*>(comp_orbits.data()), comp_orbits.size()};
    const KernelArray<PosVector> sensor_positions{const_cast<PosVector*>(SENSOR_POSITION.data()),
                                                  SENSOR_POSITION.size()};
    const KernelArray<PosVector> sensor_velocity{const_cast<PosVector*>(SENSOR_VELOCITY.data()),
                                                 SENSOR_VELOCITY.size()};

    std::vector<double> osv_lookup = CalculateOrbitStateVectorLUT(comp_orbits);

    KernelArray<double> osv_lut = {osv_lookup.data(), osv_lookup.size()};

    const GetPositionMetadata metadata{7135.669951395567, 2.3822903166873924E-8, 0.05546576,
                                       2.329562,          799303.6132771898,     sensor_positions,
                                       sensor_velocity,   orbit_state_vectors,   osv_lut};
    const auto series_size = pos_data_false.size();
    for (size_t i = 0; i < series_size; i++) {
        PositionData pos_data{};
        const auto ret = GetPosition(lats_false.at(i), lons_false.at(i), alts_false.at(i), pos_data, metadata);
        EXPECT_FALSE(ret);
        EXPECT_DOUBLE_EQ(pos_data.earth_point.x, pos_data_false.at(i).earth_point.x);
        EXPECT_DOUBLE_EQ(pos_data.earth_point.y, pos_data_false.at(i).earth_point.y);
        EXPECT_DOUBLE_EQ(pos_data.earth_point.z, pos_data_false.at(i).earth_point.z);
        EXPECT_DOUBLE_EQ(pos_data.sensor_pos.x, pos_data_false.at(i).sensor_pos.x);
        EXPECT_DOUBLE_EQ(pos_data.sensor_pos.y, pos_data_false.at(i).sensor_pos.y);
        EXPECT_DOUBLE_EQ(pos_data.sensor_pos.z, pos_data_false.at(i).sensor_pos.z);
        EXPECT_DOUBLE_EQ(pos_data.azimuth_index, pos_data_false.at(i).azimuth_index);
        EXPECT_DOUBLE_EQ(pos_data.range_index, pos_data_false.at(i).range_index);
        EXPECT_DOUBLE_EQ(pos_data.slant_range, pos_data_false.at(i).slant_range);
    }
}

TEST_F(TerrainCorrectionTest, getPositionTrueScenarioKernel) {
    std::vector<double> const lats_true{58.52938269941166, 58.52938269941166, 58.52938269941166, 58.52938269941166,
                                        58.52938269941166, 58.52938269941166, 58.52938269941166, 58.52938269941166,
                                        58.52938269941166, 58.52925774375564};
    std::vector<double> const lons_true{23.114453402942733, 23.114578358598756, 23.114703314254776, 23.114828269910795,
                                        23.11495322556682,  23.115078181222838, 23.11520313687886,  23.11532809253488,
                                        23.1154530481909,   23.113203846382525};
    std::vector<double> const alts_true{26.655392062820304, 26.766483052830047, 26.877574042861347, 26.98866503287109,
                                        27.099756022880833, 27.16567189351999,  27.204560294821366, 27.243448696122737,
                                        27.28233709742411,  25.688039075802656};
    std::vector<PositionData> const pos_data_true{{{3069968.8651965917, 1310368.109966936, 5416775.0928144},
                                                   {3658851.937123117, 1053331.0349233549, 5954284.711984713},
                                                   1499.0,
                                                   16489.057561014797,
                                                   837715.8951871425},
                                                  {{3069966.060764159, 1310374.8279885752, 5416775.187564795},
                                                   {3658851.937123117, 1053331.0349233549, 5954284.711984713},
                                                   1499.0,
                                                   16490.76257450734,
                                                   837719.8671217842},
                                                  {{3069963.2563170255, 1310381.546004214, 5416775.282315189},
                                                   {3658851.937123117, 1053331.0349233549, 5954284.711984713},
                                                   1499.0,
                                                   16492.467610722975,
                                                   837723.8391093608},
                                                  {{3069960.4518551915, 1310388.2640138534, 5416775.377065584},
                                                   {3658851.937123117, 1053331.0349233549, 5954284.711984713},
                                                   1499.0,
                                                   16494.172669660762,
                                                   837727.81114987},
                                                  {{3069957.6473786565, 1310394.9820174929, 5416775.471815978},
                                                   {3658851.937123117, 1053331.0349233549, 5954284.711984713},
                                                   1499.0,
                                                   16495.87775132104,
                                                   837731.7832433127},
                                                  {{3069954.8211966213, 1310401.6907564746, 5416775.528036151},
                                                   {3658851.9375350126, 1053331.0352269127, 5954284.711678611},
                                                   1498.9999618226757,
                                                   16497.598793672605,
                                                   837735.7925181753},
                                                  {{3069951.9820227185, 1310408.3939500444, 5416775.56120438},
                                                   {3658851.9375350126, 1053331.0352269127, 5954284.711678611},
                                                   1498.9999618226757,
                                                   16499.329393917273,
                                                   837739.8240587425},
                                                  {{3069949.1428341805, 1310415.0971374626, 5416775.594372609},
                                                   {3658851.937946909, 1053331.0355304708, 5954284.7113725105},
                                                   1498.9999236453511,
                                                   16501.060016592295,
                                                   837743.8556515626},
                                                  {{3069946.303631006, 1310421.8003187296, 5416775.627540837},
                                                   {3658851.937946909, 1053331.0355304708, 5954284.7113725105},
                                                   1498.9999236453511,
                                                   16502.790661701973,
                                                   837747.8872966456},
                                                  {{3070007.895933579, 1310305.6188605186, 5416767.001437339},
                                                   {3658851.937123117, 1053331.0349233549, 5954284.711984713},
                                                   1499.0,
                                                   16471.278523574896,
                                                   837674.477817126}};

    GetPositionMetadata metadata{
        7135.669951395567, 2.3822903166873924E-8, 0.05546576, 2.329562, 799303.6132771898, {}, {}, {}, {}};  // NOLINT

    std::vector<OrbitStateVectorComputation> comp_orbits;
    comp_orbits.reserve(ORBIT_STATE_VECTORS.size());
    for (auto&& o : ORBIT_STATE_VECTORS) {
        comp_orbits.push_back({o.time_mjd_, o.x_pos_, o.y_pos_, o.z_pos_, o.x_vel_, o.y_vel_, o.z_vel_});
    }

    const auto series_size = pos_data_true.size();
    std::vector<PositionData> position_results(series_size);
    std::vector<bool> success_results(series_size);
    LaunchGetPositionKernel(lats_true, lons_true, alts_true, position_results, metadata, SENSOR_POSITION,
                            SENSOR_VELOCITY, comp_orbits, success_results);

    CHECK_CUDA_ERR(cudaGetLastError());

    for (size_t i = 0; i < series_size; i++) {
        EXPECT_TRUE(success_results.at(i));
        EXPECT_DOUBLE_EQ(position_results.at(i).earth_point.x, pos_data_true.at(i).earth_point.x);
        EXPECT_DOUBLE_EQ(position_results.at(i).earth_point.y, pos_data_true.at(i).earth_point.y);
        EXPECT_DOUBLE_EQ(position_results.at(i).earth_point.z, pos_data_true.at(i).earth_point.z);
        EXPECT_DOUBLE_EQ(position_results.at(i).sensor_pos.x, pos_data_true.at(i).sensor_pos.x);
        EXPECT_DOUBLE_EQ(position_results.at(i).sensor_pos.y, pos_data_true.at(i).sensor_pos.y);
        EXPECT_DOUBLE_EQ(position_results.at(i).sensor_pos.z, pos_data_true.at(i).sensor_pos.z);
        EXPECT_DOUBLE_EQ(position_results.at(i).azimuth_index, pos_data_true.at(i).azimuth_index);
        // For some GPU computed floating point values are slightly different from CPU ones.
        EXPECT_NEAR(position_results.at(i).range_index, pos_data_true.at(i).range_index, 0.000000001000);
        EXPECT_NEAR(position_results.at(i).slant_range, pos_data_true.at(i).slant_range, 0.000000001000);
    }
}

TEST_F(TerrainCorrectionTest, CreateTargetProduct) {
    const double expected_geotransform[] = {21.908443888855807,     0.00012495565602102545, 0, 58.576428503903578, 0,
                                            -0.00012495565602102545};
    const int expected_width{13860};
    const int expected_height{2906};
    const double error_margin{1e-9};

    const std::vector<float> lat_tie_points{
        58.213176727294920, 58.223548889160156, 58.233741760253906, 58.243762969970700, 58.253620147705080,
        58.263313293457030, 58.272853851318360, 58.282241821289060, 58.291488647460940, 58.300598144531250,
        58.309570312500000, 58.318412780761720, 58.327129364013670, 58.335720062255860, 58.344196319580080,
        58.352554321289060, 58.360801696777344, 58.368938446044920, 58.376968383789060, 58.384895324707030,
        58.392719268798830, 58.249801635742190, 58.260181427001950, 58.270381927490234, 58.280406951904300,
        58.290267944335940, 58.299968719482420, 58.309513092041016, 58.318908691406250, 58.328159332275390,
        58.337272644042970, 58.346248626708984, 58.355094909667970, 58.363815307617190, 58.372413635253906,
        58.380889892578125, 58.389255523681640, 58.397502899169920, 58.405643463134766, 58.413677215576170,
        58.421607971191406, 58.429439544677734, 58.286430358886720, 58.296813964843750, 58.307022094726560,
        58.317050933837890, 58.326919555664060, 58.336624145507810, 58.346172332763670, 58.355571746826170,
        58.364830017089844, 58.373947143554690, 58.382926940917970, 58.391780853271484, 58.400505065917970,
        58.409103393554690, 58.417587280273440, 58.425952911376950, 58.434207916259766, 58.442352294921875,
        58.450389862060550, 58.458324432373050, 58.466156005859375, 58.323055267333984, 58.333450317382810,
        58.343658447265625, 58.353698730468750, 58.363567352294920, 58.373279571533200, 58.382831573486330,
        58.392238616943360, 58.401500701904300, 58.410621643066406, 58.419605255126950, 58.428462982177734,
        58.437191009521484, 58.445796966552734, 58.454284667968750, 58.462654113769530, 58.470912933349610,
        58.479061126708984, 58.487102508544920, 58.495040893554690, 58.502876281738280, 58.359683990478516,
        58.370082855224610, 58.380298614501950, 58.390342712402344, 58.400218963623050, 58.409931182861330,
        58.419494628906250, 58.428901672363280, 58.438167572021484, 58.447296142578125, 58.456287384033200,
        58.465145111083984, 58.473880767822266, 58.482490539550780, 58.490978240966800, 58.499355316162110,
        58.507614135742190, 58.515766143798830, 58.523811340332030, 58.531753540039060, 58.539592742919920,
        58.396308898925780, 58.406711578369140, 58.416934967041016, 58.426982879638670, 58.436862945556640,
        58.446586608886720, 58.456150054931640, 58.465564727783200, 58.474834442138670, 58.483966827392580,
        58.492961883544920, 58.501827239990234, 58.510562896728516, 58.519180297851560, 58.527671813964844,
        58.536052703857420, 58.544315338134766, 58.552471160888670, 58.560520172119140, 58.568466186523440,
        58.576309204101560};

    KernelArray<float> lat_tie_point_array{const_cast<float*>(lat_tie_points.data()), lat_tie_points.size()};

    TiePointGrid lat_grid{0, 0, 1163, 300, 21, 6, lat_tie_point_array.array};  // NOLINT

    const std::vector<float> lon_tie_points{
        21.985961914062500, 22.075984954833984, 22.165037155151367, 22.253160476684570, 22.340394973754883,
        22.426778793334960, 22.512342453002930, 22.597120285034180, 22.681142807006836, 22.764436721801758,
        22.847030639648438, 22.928949356079100, 23.010215759277344, 23.090854644775390, 23.170883178710938,
        23.250326156616210, 23.329200744628906, 23.407526016235350, 23.485319137573242, 23.562595367431640,
        23.639371871948242, 21.970470428466797, 22.060586929321290, 22.149732589721680, 22.237947463989258,
        22.325275421142578, 22.411746978759766, 22.497400283813477, 22.582267761230470, 22.666378021240234,
        22.749759674072266, 22.832441329956055, 22.914443969726562, 22.995796203613280, 23.076519012451172,
        23.156633377075195, 23.236160278320312, 23.315116882324220, 23.393524169921875, 23.471397399902344,
        23.548755645751953, 23.625612258911133, 21.954977035522460, 22.045188903808594, 22.134426116943360,
        22.222734451293945, 22.310153961181640, 22.396717071533203, 22.482460021972656, 22.567415237426758,
        22.651613235473633, 22.735082626342773, 22.817850112915040, 22.899940490722656, 22.981376647949220,
        23.062185287475586, 23.142381668090820, 23.221992492675780, 23.301033020019530, 23.379522323608400,
        23.457477569580078, 23.534915924072266, 23.611854553222656, 21.939485549926758, 22.029788970947266,
        22.119121551513672, 22.207523345947266, 22.295032501220703, 22.381685256958008, 22.467517852783203,
        22.552562713623047, 22.636848449707030, 22.720405578613280, 22.803258895874023, 22.885435104370117,
        22.966957092285156, 23.047849655151367, 23.128131866455078, 23.207824707031250, 23.286947250366210,
        23.365518569946290, 23.443555831909180, 23.521076202392578, 23.598094940185547, 21.923992156982422,
        22.014390945434570, 22.103816986083984, 22.192310333251953, 22.279911041259766, 22.366655349731445,
        22.452577590942383, 22.537710189819336, 22.622085571289062, 22.705728530883790, 22.788669586181640,
        22.870931625366210, 22.952537536621094, 23.033514022827150, 23.113880157470703, 23.193656921386720,
        23.272863388061523, 23.351516723632812, 23.429636001586914, 23.507238388061523, 23.584337234497070,
        21.908441543579100, 21.998935699462890, 22.088455200195312, 22.177040100097656, 22.264732360839844,
        22.351568222045900, 22.437580108642578, 22.522804260253906, 22.607265472412110, 22.690998077392578,
        22.774024963378906, 22.856372833251953, 22.938066482543945, 23.019128799438477, 23.099578857421875,
        23.179439544677734, 23.258728027343750, 23.337465286254883, 23.415666580200195, 23.493349075317383,
        23.570529937744140};

    KernelArray<float> lon_tie_point_array{const_cast<float*>(lon_tie_points.data()), lon_tie_points.size()};

    TiePointGrid lon_grid{0, 0, 1163, 300, 21, 6, lon_tie_point_array.array};  // NOLINT

    TiePointGeocoding source_geocoding(lat_grid, lon_grid);
    auto coh_dataset = Dataset<double>(COH_1_TIF);
    coh_dataset.LoadRasterBand(1);
    TerrainCorrection terrain_correction(coh_dataset.GetGdalDataset(), metadata_.value().GetMetadata(),
                                         metadata_.value().GetLatTiePointGrid(), metadata_.value().GetLonTiePointGrid(),
                                         srtm_3_model_->GetSrtmBuffersInfo(), srtm_3_tiles_length_);
    auto target = terrain_correction.CreateTargetProduct(&source_geocoding, TC_OUTPUT);
    const int geo_transform_array_length{6};
    double target_geo_transform[geo_transform_array_length];
    target.dataset_->GetGeoTransform(target_geo_transform);

    EXPECT_EQ(expected_height, target.dataset_->GetRasterYSize());
    EXPECT_EQ(expected_width, target.dataset_->GetRasterXSize());
    for (int i = 0; i < geo_transform_array_length; ++i) {
        EXPECT_NEAR(expected_geotransform[i], target_geo_transform[i], error_margin);
    }
    remove(TC_OUTPUT.c_str());
}

TEST_F(TerrainCorrectionTest, MetadataConstructionSucceedsOnValidFiles) {
    const std::string main_metadata_file{
        "goods/S1A_IW_SLC__1SDV_20190715T160437_20190715T160504_028130_032D5B_58D6_Orb_Stack_coh_deb.dim"};
    const std::string lat_tie_points_file{
        "goods/S1A_IW_SLC__1SDV_20190715T160437_20190715T160504_028130_032D5B_58D6_Orb_Stack_coh_deb"
        ".data/tie_point_grids/latitude.img"};
    const std::string lon_tie_points_file{
        "goods/S1A_IW_SLC__1SDV_20190715T160437_20190715T160504_028130_032D5B_58D6_Orb_Stack_coh_deb"
        ".data/tie_point_grids/longitude.img"};

    ASSERT_NO_THROW(Metadata(main_metadata_file, lat_tie_points_file, lon_tie_points_file));
}

TEST_F(TerrainCorrectionTest, MetadataConstructionThrowsWhenConstructedWithInvalidFiles) {
    const std::string main_metadata_file{
        "goods/S1A_IW_SLC__1SDV_20190715T160437_20190715T160504_028130_032D5B_58D6_Orb_Stack_coh_deb.dim"};
    const std::string lat_tie_points_file{
        "goods/S1A_IW_SLC__1SDV_20190715T160437_20190715T160504_028130_032D5B_58D6_Orb_Stack_coh_deb"
        ".data/tie_point_grids/latitude.img"};
    const std::string lon_tie_points_file{
        "goods/S1A_IW_SLC__1SDV_20190715T160437_20190715T160504_028130_032D5B_58D6_Orb_Stack_coh_deb"
        ".data/tie_point_grids/longitude.img"};

    auto gdal_error_count{0};
    alus::gdalmanagement::ErrorCallback gdal_error_count_callback = [&gdal_error_count](std::string_view) {
        gdal_error_count++;
    };
    {
        auto guard = alus::gdalmanagement::SetErrorHandle(gdal_error_count_callback);
        EXPECT_THROW(Metadata("invalid file", lat_tie_points_file, lon_tie_points_file), std::runtime_error);
        EXPECT_THROW(Metadata(main_metadata_file, "invalid lat", lon_tie_points_file), std::runtime_error);
        EXPECT_THROW(Metadata(main_metadata_file, lat_tie_points_file, "invalid_lon"), std::runtime_error);
    }
    EXPECT_EQ(gdal_error_count, 2);
}

}  // namespace
