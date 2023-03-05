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

void BackgeocodingController::BackgeocodingWorker::Work(ComputeCtx* ctx) {

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
        //std::vector<int16_t> master_tile_i(tile_size);
        //std::vector<int16_t> master_tile_q(tile_size);
        //controller_->ReadMaster(params_.master_input_area, master_tile_i.data(), master_tile_q.data());

        // TODO master conversion int16->float could be done in gdal, or possibly move to cint16 type
        //std::vector<float> out_master_tile_i(master_tile_i.begin(), master_tile_i.end());
        //std::vector<float> out_master_tile_q(master_tile_q.begin(), master_tile_q.end());

        //cuda::CudaPtr<double> device_x_points(tile_size);
        //cuda::CudaPtr<double> device_y_points(tile_size);

        //core_params.device_x_points = device_x_points.Get();
        //core_params.device_y_points = device_y_points.Get();

        core_params.stream = ctx->stream;
        core_params.device_x_points = ctx->device_x_points.EnsureBuffer<double>(tile_size);
        core_params.device_y_points = ctx->device_y_points.EnsureBuffer<double>(tile_size);

        PositionComputeResults pos_results = controller_->PositionCompute(
            params_.master_burst_index, params_.slave_burst_index, params_.master_input_area,
            core_params.device_x_points, core_params.device_y_points, ctx);


        //CHECK_CUDA_ERR(cudaFree(device_abc));
        //CHECK_CUDA_ERR(cudaFree(selected_triangles));
        //CHECK_CUDA_ERR(cudaFree(amount_of_triangles));
        //CHECK_CUDA_ERR(cudaFree(device_dtos));


        //CHECK_CUDA_ERR(cudaFree(device_zdata));
        //CHECK_CUDA_ERR(cudaFree(device_triangles));

        //ctx->device_triangles.Free();

        //CHECK_CUDA_ERR(cudaFree(device_lat_array));
        //CHECK_CUDA_ERR(cudaFree(device_lon_array));

        //CHECK_CUDA_ERR(cudaFree(calc_data.device_master_az));
        //CHECK_CUDA_ERR(cudaFree(calc_data.device_master_rg));
        //CHECK_CUDA_ERR(cudaFree(calc_data.device_slave_az));
        //CHECK_CUDA_ERR(cudaFree(calc_data.device_slave_rg));
        //CHECK_CUDA_ERR(cudaFree(calc_data.device_lats));
        //CHECK_CUDA_ERR(cudaFree(calc_data.device_lons));
        //CHECK_CUDA_ERR(cudaFree(calc_data.device_valid_index_counter));



        //CHECK_CUDA_ERR(cudaFree(device_lat_array));
        //CHECK_CUDA_ERR(cudaFree(device_lon_array));

        // Position computation succeeded
        if (pos_results.slave_area.width != 0 && pos_results.slave_area.height != 0) {
#if 0
            printf("DEMOD SZ = %zu\n", pos_results.demod_size);
            printf("DEMOD INT16 = %zu\nDEMOD DOUBLE = %zu\n", pos_results.demod_size * 2, pos_results.demod_size * 8);
            printf("TILE SZ = %zu, float = %zu\n", tile_size, tile_size * 4);
            printf("BZ :device_master_az = %zu\n", ctx->device_master_az.GetByteSize());
            printf("dev lat array = %zu\n", ctx->device_lat_array.GetByteSize());
            printf("dev lon array = %zu\n", ctx->device_lon_array.GetByteSize());
            printf("device_triangles : %zu\n", ctx->device_triangles.GetByteSize());
            printf("device_abc : %zu\n", ctx->device_abc.GetByteSize());
            printf("selected_triangles : %zu\n", ctx->selected_triangles.GetByteSize());
            printf("device_dtos : %zu\n", ctx->device_dtos.GetByteSize());
            printf("device_triangles : %zu\n", ctx->device_triangles.GetByteSize());
#endif

            std::vector<int16_t> slave_tile_i(pos_results.demod_size);
            std::vector<int16_t> slave_tile_q(pos_results.demod_size);

            controller_->ReadSlave(pos_results.slave_area, slave_tile_i.data(), slave_tile_q.data());


            core_params.device_slave_i = ctx->device_master_az.EnsureBuffer<int16_t>(pos_results.demod_size);
            core_params.device_slave_q = ctx->device_master_rg.EnsureBuffer<int16_t>(pos_results.demod_size);

            CHECK_CUDA_ERR(cudaMemcpy(core_params.device_slave_i, slave_tile_i.data(),
                                      pos_results.demod_size * sizeof(int16_t), cudaMemcpyHostToDevice));
            CHECK_CUDA_ERR(cudaMemcpy(core_params.device_slave_q, slave_tile_q.data(),
                                      pos_results.demod_size * sizeof(int16_t), cudaMemcpyHostToDevice));

            core_params.slave_rectangle = pos_results.slave_area;
            core_params.s_burst_index = params_.slave_burst_index;
            core_params.target_area = params_.master_input_area;

            //cuda::CudaPtr<double> device_demod_i(pos_results.demod_size);
            //cuda::CudaPtr<double> device_demod_q(pos_results.demod_size);
            //cuda::CudaPtr<double> device_demod_phase(pos_results.demod_size);
            core_params.device_demod_i = ctx->device_slave_az.EnsureBuffer<double>(pos_results.demod_size);
            core_params.device_demod_q = ctx->device_slave_rg.EnsureBuffer<double>(pos_results.demod_size);
            core_params.device_demod_phase = ctx->device_lats.EnsureBuffer<double>(pos_results.demod_size);

            //cuda::CudaPtr<float> device_i_results(tile_size);
            //cuda::CudaPtr<float> device_q_results(tile_size);
            core_params.device_i_results = ctx->device_lat_array.EnsureBuffer<float>(tile_size);
            core_params.device_q_results = ctx->device_lon_array.EnsureBuffer<float>(tile_size);

            controller_->CoreCompute(core_params);
            LOGV << "all computations ended: " << params_.index;
            //device_slave_i.free();
            //device_slave_q.free();
            //device_demod_i.free();
            //device_demod_q.free();
            //device_demod_phase.free();

            i_results.resize(tile_size);
            q_results.resize(tile_size);

            CHECK_CUDA_ERR(cudaMemcpy(i_results.data(), core_params.device_i_results, tile_size * sizeof(float),
                                      cudaMemcpyDeviceToHost));
            CHECK_CUDA_ERR(cudaMemcpy(q_results.data(), core_params.device_q_results, tile_size * sizeof(float),
                                      cudaMemcpyDeviceToHost));

            //device_i_results.free();
            //device_q_results.free();

        } else {  // position computation failed, write out no data values.
            i_results.resize(tile_size, controller_->output_no_data_value_);
            q_results.resize(tile_size, controller_->output_no_data_value_);
        }

        //device_x_points.free();
        //device_y_points.free();

        controller_->WriteOutputs(params_.master_input_area, nullptr, nullptr,
                                  i_results.data(), q_results.data());
    } catch (const std::exception& e) {
        LOGE << "Thread nr " << params_.index << " has caught exception " << e.what();
        controller_->RegisterException(std::current_exception());
    }


    LOGV << "Death of worker: " << params_.index;
}

}  // namespace alus::backgeocoding
