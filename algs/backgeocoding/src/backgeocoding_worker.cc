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

namespace alus {
namespace backgeocoding {

BackgeocodingController::BackgeocodingWorker::~BackgeocodingWorker() {
    std::cout << "Death of worker: " << params_.index << std::endl;
}

void BackgeocodingController::BackgeocodingWorker::Work() {
    CoreComputeParams core_params;
    std::vector<float> i_results;
    std::vector<float> q_results;

    std::unique_lock<std::mutex> lk(controller_->queue_mutex_);
    controller_->worker_counter_++; //mark down that this thread is now fully functional and started
    controller_->GetThreadSync()->wait(lk);

    std::cout << "I am now working. Worker: " << params_.index << std::endl;

    try{
        const size_t tile_size = params_.master_input_area.width * params_.master_input_area.height;
        std::vector<double> master_tile_i(tile_size);
        std::vector<double> master_tile_q(tile_size);
        controller_->ReadMaster(params_.master_input_area, master_tile_i.data(), master_tile_q.data());

        CHECK_CUDA_ERR(cudaMalloc((void **)&core_params.device_x_points, tile_size * sizeof(double)));
        CHECK_CUDA_ERR(cudaMalloc((void **)&core_params.device_y_points, tile_size * sizeof(double)));

        PositionComputeResults pos_results = controller_->PositionCompute(params_.master_burst_index,
                                                                          params_.slave_burst_index,
                                                                          params_.master_input_area,
                                                                          core_params.device_x_points,
                                                                          core_params.device_y_points);
        //Position computation succeeded
        if(pos_results.slave_area.width != 0 && pos_results.slave_area.height != 0){
            std::vector<double> slave_tile_i(pos_results.demod_size);
            std::vector<double> slave_tile_q(pos_results.demod_size);

            controller_->ReadSlave(pos_results.slave_area, slave_tile_i.data(), slave_tile_q.data());

            CHECK_CUDA_ERR(cudaMalloc((void **)&core_params.device_slave_i, pos_results.demod_size * sizeof(double)));
            CHECK_CUDA_ERR(cudaMalloc((void **)&core_params.device_slave_q, pos_results.demod_size * sizeof(double)));

            CHECK_CUDA_ERR(cudaMemcpy(core_params.device_slave_i,
                                      slave_tile_i.data(),
                                      pos_results.demod_size * sizeof(double),
                                      cudaMemcpyHostToDevice));
            CHECK_CUDA_ERR(cudaMemcpy(core_params.device_slave_q,
                                      slave_tile_q.data(),
                                      pos_results.demod_size * sizeof(double),
                                      cudaMemcpyHostToDevice));

            core_params.slave_rectangle = pos_results.slave_area;
            core_params.s_burst_index = params_.slave_burst_index;
            core_params.target_area = params_.master_input_area;

            CHECK_CUDA_ERR(cudaMalloc((void **)&core_params.device_demod_i, pos_results.demod_size * sizeof(double)));
            CHECK_CUDA_ERR(cudaMalloc((void **)&core_params.device_demod_q, pos_results.demod_size * sizeof(double)));
            CHECK_CUDA_ERR(cudaMalloc((void **)&core_params.device_demod_phase, pos_results.demod_size * sizeof(double)));

            CHECK_CUDA_ERR(cudaMalloc((void **)&core_params.device_i_results, tile_size * sizeof(float)));
            CHECK_CUDA_ERR(cudaMalloc((void **)&core_params.device_q_results, tile_size * sizeof(float)));
            CHECK_CUDA_ERR(cudaMemset(core_params.device_i_results, 0, tile_size * sizeof(float)));
            CHECK_CUDA_ERR(cudaMemset(core_params.device_q_results, 0, tile_size * sizeof(float)));

            controller_->CoreCompute(core_params);
            CHECK_CUDA_ERR(cudaFree(core_params.device_slave_i));
            CHECK_CUDA_ERR(cudaFree(core_params.device_slave_q));
            CHECK_CUDA_ERR(cudaFree(core_params.device_demod_i));
            CHECK_CUDA_ERR(cudaFree(core_params.device_demod_q));
            CHECK_CUDA_ERR(cudaFree(core_params.device_demod_phase));

            i_results.resize(tile_size);
            q_results.resize(tile_size);

            CHECK_CUDA_ERR(
                cudaMemcpy(i_results.data(), core_params.device_i_results, tile_size * sizeof(float), cudaMemcpyDeviceToHost));
            CHECK_CUDA_ERR(
                cudaMemcpy(q_results.data(), core_params.device_q_results, tile_size * sizeof(float), cudaMemcpyDeviceToHost));

            CHECK_CUDA_ERR(cudaFree(core_params.device_i_results));
            CHECK_CUDA_ERR(cudaFree(core_params.device_q_results));

            controller_->WriteOutputs(params_.master_input_area, i_results.data(), q_results.data());
        }else{ //position computation failed, write out no data values.
            i_results.resize(tile_size, controller_->output_no_data_value_);
            q_results.resize(tile_size, controller_->output_no_data_value_);
        }

        CHECK_CUDA_ERR(cudaFree(core_params.device_x_points));
        CHECK_CUDA_ERR(cudaFree(core_params.device_y_points));
        controller_->WriteOutputs(params_.master_input_area, i_results.data(), q_results.data());
    } catch (const std::exception &e) {
        std::cerr << "Thread nr " <<
        params_.index << " covering master recangle "<<
        params_.master_input_area.x << " " <<
        params_.master_input_area.y << " " <<
        params_.master_input_area.width << " " <<
        params_.master_input_area.height << " has caught exception " << std::endl << e.what() << std::endl;
        controller_->RegisterThreadEnd();
        throw;
    }

    controller_->RegisterThreadEnd();
}

}  // namespace backgeocoding
}  // namespace alus