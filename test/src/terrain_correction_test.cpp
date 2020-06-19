#include <numeric>
#include <optional>

#include "gmock/gmock.h"

#include "cuda_util.hpp"
#include "get_position.h"
#include "position_data.h"
#include "terrain_correction.hpp"
#include "tests_common.hpp"

#include "../goods/S1A_IW_SLC__1SDV_20190715T160437_20190715T160504_028130_032D5B_58D6_Orb_Stack_coh_deb_orbit.hpp"

namespace alus::tests {
// Forward declare function from terrain_correction_test_kernels.cu.
void LaunchGetPositionKernel(const std::vector<double>& lat_args,
                             const std::vector<double>& lon_args,
                             const std::vector<double>& alt_args,
                             std::vector<terraincorrection::PositionData>& sat_positions,
                             terraincorrection::GetPositionMetadata metadata,
                             const std::vector<snapengine::PosVector>& sensor_position,
                             const std::vector<snapengine::PosVector>& sensor_velocity,
                             const std::vector<snapengine::OrbitStateVector>& orbit_state_vector,
                             std::vector<bool>& results);
}

namespace {

using namespace alus::cudautil;
using namespace alus::goods;
using namespace alus::tests;
using namespace alus::terraincorrection;
using namespace alus::snapengine;

class TerrainCorrectionTest : public ::testing::Test {
   public:
    TerrainCorrectionTest() {
        coh_ds_ = std::make_optional<alus::Dataset>(COH_1_TIF);
        coh_ds_.value().loadRasterBand(1);
        dem_ds_ = std::make_optional<alus::Dataset>(DEM_PATH_1);
        dem_ds_.value().loadRasterBand(1);
    }

    std::optional<alus::Dataset> coh_ds_;
    std::optional<alus::Dataset> coh_data_ds_;
    std::optional<alus::Dataset> dem_ds_;

   protected:
};

TEST_F(TerrainCorrectionTest, fetchElevationsOnGPU) {
    alus::TerrainCorrection tc{std::move(coh_ds_.value()), std::move(coh_ds_.value()), std::move(dem_ds_.value())};
    tc.localDemCuda();
    const auto& elevations = tc.getElevations();
    const auto [min, max] = std::minmax_element(std::begin(elevations), std::end(elevations));
    EXPECT_EQ(*min, 0);
    EXPECT_EQ(*max, 43);
    auto const avg = std::accumulate(elevations.cbegin(), elevations.cend(), 0.0) / elevations.size();
    EXPECT_DOUBLE_EQ(avg, 2.960957384655039);
}

TEST_F(TerrainCorrectionTest, getPositionTrueScenario) {
    std::vector<double> const LATS_TRUE{58.52938269941166,
                                        58.52938269941166,
                                        58.52938269941166,
                                        58.52938269941166,
                                        58.52938269941166,
                                        58.52938269941166,
                                        58.52938269941166,
                                        58.52938269941166,
                                        58.52938269941166,
                                        58.52925774375564};
    std::vector<double> const LONS_TRUE{23.114453402942733,
                                        23.114578358598756,
                                        23.114703314254776,
                                        23.114828269910795,
                                        23.11495322556682,
                                        23.115078181222838,
                                        23.11520313687886,
                                        23.11532809253488,
                                        23.1154530481909,
                                        23.113203846382525};
    std::vector<double> const ALTS_TRUE{26.655392062820304,
                                        26.766483052830047,
                                        26.877574042861347,
                                        26.98866503287109,
                                        27.099756022880833,
                                        27.16567189351999,
                                        27.204560294821366,
                                        27.243448696122737,
                                        27.28233709742411,
                                        25.688039075802656};
    std::vector<PositionData> const POS_DATA_TRUE{{{3069968.8651965917, 1310368.109966936, 5416775.0928144},
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

    const KernelArray<OrbitStateVector> orbitStateVectors{const_cast<OrbitStateVector*>(ORBIT_STATE_VECTORS.data()),
                                                          ORBIT_STATE_VECTORS.size()};
    const KernelArray<PosVector> sensorPositions{const_cast<PosVector*>(SENSOR_POSITION.data()),
                                                 SENSOR_POSITION.size()};
    const KernelArray<PosVector> sensorVelocity{const_cast<PosVector*>(SENSOR_VELOCITY.data()), SENSOR_VELOCITY.size()};

    const GetPositionMetadata metadata{7135.669951395567,
                                       2.3822903166873924E-8,
                                       0.05546576,
                                       2.329562,
                                       799303.6132771898,
                                       sensorPositions,
                                       sensorVelocity,
                                       orbitStateVectors};
    const auto series_size = POS_DATA_TRUE.size();
    for (size_t i = 0; i < series_size; i++) {
        PositionData pos_data{};
        const auto ret = GetPosition(LATS_TRUE.at(i), LONS_TRUE.at(i), ALTS_TRUE.at(i), pos_data, metadata);
        EXPECT_TRUE(ret);
        EXPECT_DOUBLE_EQ(pos_data.earth_point.x, POS_DATA_TRUE.at(i).earth_point.x);
        EXPECT_DOUBLE_EQ(pos_data.earth_point.y, POS_DATA_TRUE.at(i).earth_point.y);
        EXPECT_DOUBLE_EQ(pos_data.earth_point.z, POS_DATA_TRUE.at(i).earth_point.z);
        EXPECT_DOUBLE_EQ(pos_data.sensor_pos.x, POS_DATA_TRUE.at(i).sensor_pos.x);
        EXPECT_DOUBLE_EQ(pos_data.sensor_pos.y, POS_DATA_TRUE.at(i).sensor_pos.y);
        EXPECT_DOUBLE_EQ(pos_data.sensor_pos.z, POS_DATA_TRUE.at(i).sensor_pos.z);
        EXPECT_DOUBLE_EQ(pos_data.azimuth_index, POS_DATA_TRUE.at(i).azimuth_index);
        EXPECT_DOUBLE_EQ(pos_data.range_index, POS_DATA_TRUE.at(i).range_index);
        EXPECT_DOUBLE_EQ(pos_data.slant_range, POS_DATA_TRUE.at(i).slant_range);
    }
}

TEST_F(TerrainCorrectionTest, getPositionFalseScenario) {
    std::vector<double> const LATS_FALSE{58.57074302155462,
                                         58.57074302155462,
                                         58.57074302155462,
                                         58.57074302155462,
                                         58.57074302155462,
                                         58.57074302155462,
                                         58.57074302155462,
                                         58.57074302155462,
                                         58.57074302155462,
                                         58.57074302155462};
    std::vector<double> const LONS_FALSE{22.29012094017203,
                                         22.29024589582805,
                                         22.29037085148407,
                                         22.290495807140093,
                                         22.290620762796113,
                                         22.290745718452133,
                                         22.290870674108156,
                                         22.290995629764176,
                                         22.2911205854202,
                                         22.29124554107622};
    std::vector<double> const ALTS_FALSE{32.416427961115495,
                                         32.38403161384701,
                                         32.3516352665848,
                                         32.319238919316305,
                                         32.286842572054105,
                                         32.2544462247919,
                                         32.182097076725086,
                                         32.01600429552241,
                                         31.849911514287488,
                                         31.683818733084806};
    std::vector<PositionData> const POS_DATA_FALSE{
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

    const KernelArray<OrbitStateVector> orbitStateVectors{const_cast<OrbitStateVector*>(ORBIT_STATE_VECTORS.data()),
                                                          ORBIT_STATE_VECTORS.size()};
    const KernelArray<PosVector> sensorPositions{const_cast<PosVector*>(SENSOR_POSITION.data()),
                                                 SENSOR_POSITION.size()};
    const KernelArray<PosVector> sensorVelocity{const_cast<PosVector*>(SENSOR_VELOCITY.data()), SENSOR_VELOCITY.size()};

    const GetPositionMetadata metadata{7135.669951395567,
                                       2.3822903166873924E-8,
                                       0.05546576,
                                       2.329562,
                                       799303.6132771898,
                                       sensorPositions,
                                       sensorVelocity,
                                       orbitStateVectors};
    const auto series_size = POS_DATA_FALSE.size();
    for (size_t i = 0; i < series_size; i++) {
        PositionData pos_data{};
        const auto ret = GetPosition(LATS_FALSE.at(i), LONS_FALSE.at(i), ALTS_FALSE.at(i), pos_data, metadata);
        EXPECT_FALSE(ret);
        EXPECT_DOUBLE_EQ(pos_data.earth_point.x, POS_DATA_FALSE.at(i).earth_point.x);
        EXPECT_DOUBLE_EQ(pos_data.earth_point.y, POS_DATA_FALSE.at(i).earth_point.y);
        EXPECT_DOUBLE_EQ(pos_data.earth_point.z, POS_DATA_FALSE.at(i).earth_point.z);
        EXPECT_DOUBLE_EQ(pos_data.sensor_pos.x, POS_DATA_FALSE.at(i).sensor_pos.x);
        EXPECT_DOUBLE_EQ(pos_data.sensor_pos.y, POS_DATA_FALSE.at(i).sensor_pos.y);
        EXPECT_DOUBLE_EQ(pos_data.sensor_pos.z, POS_DATA_FALSE.at(i).sensor_pos.z);
        EXPECT_DOUBLE_EQ(pos_data.azimuth_index, POS_DATA_FALSE.at(i).azimuth_index);
        EXPECT_DOUBLE_EQ(pos_data.range_index, POS_DATA_FALSE.at(i).range_index);
        EXPECT_DOUBLE_EQ(pos_data.slant_range, POS_DATA_FALSE.at(i).slant_range);
    }
}

TEST_F(TerrainCorrectionTest, getPositionTrueScenarioKernel) {
    std::vector<double> const LATS_TRUE{58.52938269941166,
                                        58.52938269941166,
                                        58.52938269941166,
                                        58.52938269941166,
                                        58.52938269941166,
                                        58.52938269941166,
                                        58.52938269941166,
                                        58.52938269941166,
                                        58.52938269941166,
                                        58.52925774375564};
    std::vector<double> const LONS_TRUE{23.114453402942733,
                                        23.114578358598756,
                                        23.114703314254776,
                                        23.114828269910795,
                                        23.11495322556682,
                                        23.115078181222838,
                                        23.11520313687886,
                                        23.11532809253488,
                                        23.1154530481909,
                                        23.113203846382525};
    std::vector<double> const ALTS_TRUE{26.655392062820304,
                                        26.766483052830047,
                                        26.877574042861347,
                                        26.98866503287109,
                                        27.099756022880833,
                                        27.16567189351999,
                                        27.204560294821366,
                                        27.243448696122737,
                                        27.28233709742411,
                                        25.688039075802656};
    std::vector<PositionData> const POS_DATA_TRUE{{{3069968.8651965917, 1310368.109966936, 5416775.0928144},
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

    const GetPositionMetadata metadata{
        7135.669951395567, 2.3822903166873924E-8, 0.05546576, 2.329562, 799303.6132771898, {}, {}, {}};

    const auto series_size = POS_DATA_TRUE.size();
    std::vector<PositionData> positionResults(series_size);
    std::vector<bool> successResults(series_size);
    LaunchGetPositionKernel(LATS_TRUE,
                            LONS_TRUE,
                            ALTS_TRUE,
                            positionResults,
                            metadata,
                            SENSOR_POSITION,
                            SENSOR_VELOCITY,
                            ORBIT_STATE_VECTORS,
                            successResults);

    CHECK_CUDA_ERR(cudaGetLastError());

    for (size_t i = 0; i < series_size; i++) {
        EXPECT_TRUE(successResults.at(i));
        EXPECT_DOUBLE_EQ(positionResults.at(i).earth_point.x, POS_DATA_TRUE.at(i).earth_point.x);
        EXPECT_DOUBLE_EQ(positionResults.at(i).earth_point.y, POS_DATA_TRUE.at(i).earth_point.y);
        EXPECT_DOUBLE_EQ(positionResults.at(i).earth_point.z, POS_DATA_TRUE.at(i).earth_point.z);
        EXPECT_DOUBLE_EQ(positionResults.at(i).sensor_pos.x, POS_DATA_TRUE.at(i).sensor_pos.x);
        EXPECT_DOUBLE_EQ(positionResults.at(i).sensor_pos.y, POS_DATA_TRUE.at(i).sensor_pos.y);
        EXPECT_DOUBLE_EQ(positionResults.at(i).sensor_pos.z, POS_DATA_TRUE.at(i).sensor_pos.z);
        EXPECT_DOUBLE_EQ(positionResults.at(i).azimuth_index, POS_DATA_TRUE.at(i).azimuth_index);
        // For some GPU computed floating point values are slightly different from CPU ones.
        EXPECT_NEAR(positionResults.at(i).range_index, POS_DATA_TRUE.at(i).range_index, 0.000000001000);
        EXPECT_NEAR(positionResults.at(i).slant_range, POS_DATA_TRUE.at(i).slant_range, 0.000000001000);
    }
}

}  // namespace
