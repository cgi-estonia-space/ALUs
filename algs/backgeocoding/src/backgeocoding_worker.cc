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
#include "backgeocoding_controller.h"

#include <vector>

#include "alus_log.h"
#include "cuda_ptr.h"
#include "cuda_util.h"

namespace alus::backgeocoding {

void BackgeocodingController::BackgeocodingWorker::Work() {
    CoreComputeParams core_params;
    std::vector<float> i_results;
    std::vector<float> q_results;

    if (!controller_->exceptions_thrown_) {
        LOGV << "I am now working. Worker: " << params_.index << " rect = [" << params_.master_input_area.x << " "
             << params_.master_input_area.y << " " << params_.master_input_area.width << " "
             << params_.master_input_area.height << "]";
    } else {
        LOGW << "Worker " << params_.index
             << " has detected that exceptions were thrown elsewhere and is shutting down.";
        return;
    }

    try {
        const size_t tile_size = params_.master_input_area.width * params_.master_input_area.height;
        std::vector<int16_t> master_tile_i(tile_size);
        std::vector<int16_t> master_tile_q(tile_size);
        controller_->ReadMaster(params_.master_input_area, master_tile_i.data(), master_tile_q.data());

        // TODO master conversion int16->float could be done in gdal, or possibly move to cint16 type
        std::vector<float> out_master_tile_i(master_tile_i.begin(), master_tile_i.end());
        std::vector<float> out_master_tile_q(master_tile_q.begin(), master_tile_q.end());

        cuda::CudaPtr<double> device_x_points(tile_size);
        cuda::CudaPtr<double> device_y_points(tile_size);
        core_params.device_x_points = device_x_points.Get();
        core_params.device_y_points = device_y_points.Get();

        PositionComputeResults pos_results = controller_->PositionCompute(
            params_.master_burst_index, params_.slave_burst_index, params_.master_input_area,
            core_params.device_x_points, core_params.device_y_points);

        // Position computation succeeded
        if (pos_results.slave_area.width != 0 && pos_results.slave_area.height != 0) {
            std::vector<int16_t> slave_tile_i(pos_results.demod_size);
            std::vector<int16_t> slave_tile_q(pos_results.demod_size);

            controller_->ReadSlave(pos_results.slave_area, slave_tile_i.data(), slave_tile_q.data());

            cuda::CudaPtr<int16_t> device_slave_i(pos_results.demod_size);
            cuda::CudaPtr<int16_t> device_slave_q(pos_results.demod_size);
            core_params.device_slave_i = device_slave_i.Get();
            core_params.device_slave_q = device_slave_q.Get();

            CHECK_CUDA_ERR(cudaMemcpy(core_params.device_slave_i, slave_tile_i.data(),
                                      pos_results.demod_size * sizeof(int16_t), cudaMemcpyHostToDevice));
            CHECK_CUDA_ERR(cudaMemcpy(core_params.device_slave_q, slave_tile_q.data(),
                                      pos_results.demod_size * sizeof(int16_t), cudaMemcpyHostToDevice));

            core_params.slave_rectangle = pos_results.slave_area;
            core_params.s_burst_index = params_.slave_burst_index;
            core_params.target_area = params_.master_input_area;

            cuda::CudaPtr<double> device_demod_i(pos_results.demod_size);
            cuda::CudaPtr<double> device_demod_q(pos_results.demod_size);
            cuda::CudaPtr<double> device_demod_phase(pos_results.demod_size);
            core_params.device_demod_i = device_demod_i.Get();
            core_params.device_demod_q = device_demod_q.Get();
            core_params.device_demod_phase = device_demod_phase.Get();

            cuda::CudaPtr<float> device_i_results(tile_size);
            cuda::CudaPtr<float> device_q_results(tile_size);
            core_params.device_i_results = device_i_results.Get();
            core_params.device_q_results = device_q_results.Get();

            controller_->CoreCompute(core_params);
            LOGV << "all computations ended: " << params_.index;
            device_slave_i.free();
            device_slave_q.free();
            device_demod_i.free();
            device_demod_q.free();
            device_demod_phase.free();

            i_results.resize(tile_size);
            q_results.resize(tile_size);

            CHECK_CUDA_ERR(cudaMemcpy(i_results.data(), core_params.device_i_results, tile_size * sizeof(float),
                                      cudaMemcpyDeviceToHost));
            CHECK_CUDA_ERR(cudaMemcpy(q_results.data(), core_params.device_q_results, tile_size * sizeof(float),
                                      cudaMemcpyDeviceToHost));

            device_i_results.free();
            device_q_results.free();

        } else {  // position computation failed, write out no data values.
            i_results.resize(tile_size, controller_->output_no_data_value_);
            q_results.resize(tile_size, controller_->output_no_data_value_);
        }

        device_x_points.free();
        device_y_points.free();
        controller_->WriteOutputs(params_.master_input_area, out_master_tile_i.data(), out_master_tile_q.data(),
                                  i_results.data(), q_results.data());
    } catch (const std::exception& e) {
        LOGE << "Thread nr " << params_.index << " has caught exception " << e.what();
        controller_->RegisterException(std::current_exception());
    }
    LOGV << "Death of worker: " << params_.index;
}

}  // namespace alus::backgeocoding
