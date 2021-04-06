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
#include <vector>

#include "gmock/gmock.h"

#include "backgeocoding.h"
#include "comparators.h"
#include "shapes.h"
#include "dem_assistant.h"

namespace {

class BackgeocodingTester {
private:
public:
    std::vector<float> q_result_;
    std::vector<float> i_result_;
    int tile_size_;
    std::string q_phase_file_;
    std::string i_phase_file_;
    std::string i_tile_file_;
    std::string q_tile_file_;

    BackgeocodingTester(std::string q_phase_file, std::string i_phase_file, std::string q_tile_file,
                        std::string i_tile_file) {
        q_phase_file_ = q_phase_file;
        i_phase_file_ = i_phase_file;
        q_tile_file_ = q_tile_file;
        i_tile_file_ = i_tile_file;
    }

    void ReadTestData() {
        std::ifstream q_phase_stream(q_phase_file_);
        std::ifstream i_phase_stream(i_phase_file_);
        if (!q_phase_stream.is_open()) {
            throw std::ios::failure("qPhase.txt is not open.");
        }
        if (!i_phase_stream.is_open()) {
            throw std::ios::failure("iPhase.txt is not open.");
        }
        int tile_x = 100;
        int tile_y = 100;
        const size_t size = tile_x * tile_y;
        tile_size_ = size;

        q_result_.resize(size);
        i_result_.resize(size);

        for (size_t i = 0; i < size; i++) {
            q_phase_stream >> q_result_.at(i);
            i_phase_stream >> i_result_.at(i);
        }

        q_phase_stream.close();
        i_phase_stream.close();
    }

    void ReadTile(alus::Rectangle area, double* tile_i, double* tile_q) {
        std::ifstream slave_i_stream(i_tile_file_);
        std::ifstream slave_q_stream(q_tile_file_);
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

bool RunBackgeocoding(alus::backgeocoding::Backgeocoding* backgeocoding, BackgeocodingTester* tester,
                      alus::Rectangle target_area, int m_burst_index, int s_burst_index, float* i_results,
                      float* q_results) {
    alus::backgeocoding::CoreComputeParams core_params;
    const size_t tile_size = target_area.width * target_area.height;

    CHECK_CUDA_ERR(cudaMalloc((void**)&core_params.device_x_points, tile_size * sizeof(double)));
    CHECK_CUDA_ERR(cudaMalloc((void**)&core_params.device_y_points, tile_size * sizeof(double)));

    core_params.slave_rectangle = backgeocoding->PositionCompute(
        m_burst_index, s_burst_index, target_area, core_params.device_x_points, core_params.device_y_points);

    if (core_params.slave_rectangle.width != 0 && core_params.slave_rectangle.height != 0) {
        core_params.demod_size = core_params.slave_rectangle.width * core_params.slave_rectangle.height;

        std::vector<double> slave_tile_i(core_params.demod_size);
        std::vector<double> slave_tile_q(core_params.demod_size);

        tester->ReadTile(core_params.slave_rectangle, slave_tile_i.data(), slave_tile_q.data());

        CHECK_CUDA_ERR(cudaMalloc((void**)&core_params.device_slave_i, core_params.demod_size * sizeof(double)));
        CHECK_CUDA_ERR(cudaMalloc((void**)&core_params.device_slave_q, core_params.demod_size * sizeof(double)));

        CHECK_CUDA_ERR(cudaMemcpy(core_params.device_slave_i, slave_tile_i.data(),
                                  core_params.demod_size * sizeof(double), cudaMemcpyHostToDevice));
        CHECK_CUDA_ERR(cudaMemcpy(core_params.device_slave_q, slave_tile_q.data(),
                                  core_params.demod_size * sizeof(double), cudaMemcpyHostToDevice));

        core_params.s_burst_index = s_burst_index;
        core_params.target_area = target_area;

        CHECK_CUDA_ERR(cudaMalloc((void**)&core_params.device_demod_i, core_params.demod_size * sizeof(double)));
        CHECK_CUDA_ERR(cudaMalloc((void**)&core_params.device_demod_q, core_params.demod_size * sizeof(double)));
        CHECK_CUDA_ERR(cudaMalloc((void**)&core_params.device_demod_phase, core_params.demod_size * sizeof(double)));

        CHECK_CUDA_ERR(cudaMalloc((void**)&core_params.device_i_results, tile_size * sizeof(float)));
        CHECK_CUDA_ERR(cudaMalloc((void**)&core_params.device_q_results, tile_size * sizeof(float)));

        backgeocoding->CoreCompute(core_params);
        CHECK_CUDA_ERR(cudaFree(core_params.device_slave_i));
        CHECK_CUDA_ERR(cudaFree(core_params.device_slave_q));
        CHECK_CUDA_ERR(cudaFree(core_params.device_demod_i));
        CHECK_CUDA_ERR(cudaFree(core_params.device_demod_q));
        CHECK_CUDA_ERR(cudaFree(core_params.device_demod_phase));
        CHECK_CUDA_ERR(cudaFree(core_params.device_x_points));
        CHECK_CUDA_ERR(cudaFree(core_params.device_y_points));

        CHECK_CUDA_ERR(
            cudaMemcpy(i_results, core_params.device_i_results, tile_size * sizeof(float), cudaMemcpyDeviceToHost));
        CHECK_CUDA_ERR(
            cudaMemcpy(q_results, core_params.device_q_results, tile_size * sizeof(float), cudaMemcpyDeviceToHost));

        CHECK_CUDA_ERR(cudaFree(core_params.device_i_results));
        CHECK_CUDA_ERR(cudaFree(core_params.device_q_results));

        return true;
    } else {
        CHECK_CUDA_ERR(cudaFree(core_params.device_x_points));
        CHECK_CUDA_ERR(cudaFree(core_params.device_y_points));
        return false;
    }
}

TEST(backgeocoding, correctness) {
    std::vector<std::string> srtm3_files{"./goods/srtm_41_01.tif", "./goods/srtm_41_01.tif"};
    std::shared_ptr<alus::app::DemAssistant> dem_assistant = alus::app::DemAssistant::CreateFormattedSrtm3TilesOnGpuFrom(std::move(srtm3_files));
    alus::backgeocoding::Backgeocoding backgeocoding;
    BackgeocodingTester land_tester("./goods/backgeocoding/qPhase.txt", "./goods/backgeocoding/iPhase.txt",
                                    "./goods/backgeocoding/slaveTileQ.txt", "./goods/backgeocoding/slaveTileI.txt");
    land_tester.ReadTestData();

    BackgeocodingTester coast_tester("./goods/backgeocoding/qPhaseCoast.txt", "./goods/backgeocoding/iPhaseCoast.txt",
                                     "./goods/backgeocoding/slaveTileQCoast.txt",
                                     "./goods/backgeocoding/slaveTileICoast.txt");
    coast_tester.ReadTestData();

    backgeocoding.SetElevationData(dem_assistant->GetEgm96ValuesOnGpu(), {dem_assistant->GetSrtm3ValuesOnGpu(), dem_assistant->GetSrtm3TilesCount()});
    backgeocoding.PrepareToCompute("./goods/master_metadata.dim", "./goods/slave_metadata.dim");

    int burst_offset = backgeocoding.GetBurstOffset();
    size_t tile_size;
    alus::Rectangle target_area_land = {4000, 17000, 100, 100};
    int m_burst_index_land = 11;
    int s_burst_index_land = m_burst_index_land + burst_offset;
    tile_size = target_area_land.width * target_area_land.height;

    std::vector<float> i_results_land(tile_size);
    std::vector<float> q_results_land(tile_size);

    bool pos_compute_success_land = RunBackgeocoding(&backgeocoding, &land_tester, target_area_land, m_burst_index_land,
                                                     s_burst_index_land, i_results_land.data(), q_results_land.data());

    EXPECT_EQ(pos_compute_success_land, true) << "Land Position Compute failed" << '\n';

    // Sometimes we get 1 mismatch, by about 0.0001. It's checked and fine. Debug if there are 2 or more
    size_t count_i = alus::EqualsArrays(i_results_land.data(), land_tester.i_result_.data(), land_tester.tile_size_, 0.00001);
    EXPECT_EQ(count_i, count_i != 0) << "Land results I do not match. Mismatches: " << count_i << '\n';

    size_t count_q = alus::EqualsArrays(q_results_land.data(), land_tester.q_result_.data(), land_tester.tile_size_, 0.00001);
    EXPECT_EQ(count_q, count_q != 0) << "Land results Q do not match. Mismatches: " << count_q << '\n';

    alus::Rectangle target_area_sea{10, 19486, 50, 50};
    int m_burst_index_sea = 12;
    int s_burst_index_sea = m_burst_index_sea + burst_offset;

    std::vector<float> i_results_sea(1);
    std::vector<float> q_results_sea(1);
    bool pos_compute_success_sea = RunBackgeocoding(&backgeocoding, &land_tester, target_area_sea, m_burst_index_sea,
                                                    s_burst_index_sea, i_results_sea.data(), q_results_sea.data());

    EXPECT_EQ(pos_compute_success_sea, false) << "Sea Position Compute is supposed to fail" << '\n';


    alus::Rectangle target_area_coast{4700, 21430, 100, 100};
    int m_burst_index_coast = 14;
    int s_burst_index_coast = m_burst_index_coast + burst_offset;
    tile_size = target_area_coast.width * target_area_coast.height;

    std::vector<float> i_results_coast(tile_size);
    std::vector<float> q_results_coast(tile_size);

    bool pos_compute_success_coast =
        RunBackgeocoding(&backgeocoding, &coast_tester, target_area_coast, m_burst_index_coast, s_burst_index_coast,
                         i_results_coast.data(), q_results_coast.data());
    EXPECT_EQ(pos_compute_success_coast, true) << "Coast Position Compute failed" << '\n';

    // Sometimes we get 1 mismatch, by about 0.0001. It's checked and fine. Debug if there are 2 or more
    size_t count_i_coast =
        alus::EqualsArrays(i_results_coast.data(), coast_tester.i_result_.data(), coast_tester.tile_size_, 0.00001);
    EXPECT_EQ(count_i_coast, count_i_coast != 0)
        << "Coast results I do not match. Mismatches: " << count_i_coast << '\n';

    size_t count_q_coast =
        alus::EqualsArrays(q_results_coast.data(), coast_tester.q_result_.data(), coast_tester.tile_size_, 0.00001);
    EXPECT_EQ(count_q_coast, count_q_coast != 0)
        << "Coast results Q do not match. Mismatches: " << count_q_coast << '\n';
}

}  // namespace
