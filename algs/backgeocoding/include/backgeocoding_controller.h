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
#pragma once

#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <string_view>
#include <thread>
#include <vector>

#include "alus_file_reader.h"
#include "alus_file_writer.h"
#include "backgeocoding.h"
#include "pointer_holders.h"
#include "snap-core/datamodel/product.h"

namespace alus::backgeocoding {

struct PositionComputeResults {
    Rectangle slave_area;
    size_t demod_size;
};

/**
 * A helper class to manage the data intputs and threading to Backgeocoding class.
 * IMPORTANT: input products can only have 1 subswath!
 */
class BackgeocodingController {
public:
    std::mutex queue_mutex_;
    std::atomic<size_t> worker_counter_;
    const float output_no_data_value_ = 0.0;
    int exceptions_thrown_ = 0;

    BackgeocodingController(std::shared_ptr<AlusFileReader<double>> master_input_dataset,
                            std::shared_ptr<AlusFileReader<double>> slave_input_dataset,
                            std::shared_ptr<AlusFileWriter<float>> output_dataset,
                            std::string_view master_metadata_file, std::string_view slave_metadata_file);

    BackgeocodingController(std::shared_ptr<AlusFileReader<double>> master_input_dataset,
                            std::shared_ptr<AlusFileReader<double>> slave_input_dataset,
                            std::shared_ptr<AlusFileWriter<float>> output_dataset,
                            std::shared_ptr<snapengine::Product> master_product,
                            std::shared_ptr<snapengine::Product> slave_product);

    ~BackgeocodingController();
    BackgeocodingController(const BackgeocodingController&) = delete;  // class does not support copying(and moving)
    BackgeocodingController& operator=(const BackgeocodingController&) = delete;

    void PrepareToCompute(const float* egm96_device_Array, PointerArray srtm3_tiles);
    void RegisterThreadEnd();
    void RegisterException(std::exception_ptr e);
    void ReadMaster(Rectangle master_area, double* i_tile, double* q_tile);
    PositionComputeResults PositionCompute(int m_burst_index, int s_burst_index, Rectangle target_area,
                                           double* device_x_points, double* device_y_points);
    void ReadSlave(Rectangle slave_area, double* i_tile, double* q_tile);
    void CoreCompute(CoreComputeParams params);
    void WriteOutputs(Rectangle output_area, float* i_master_results, float* q_master_results, float* i_slave_results, float* q_slave_results);
    void DoWork();
    void Initialize();

    std::condition_variable* GetThreadSync() { return &thread_sync_; }

private:
    std::unique_ptr<Backgeocoding> backgeocoding_;
    std::vector<std::exception_ptr> exceptions_;
    int num_of_bursts_;
    int lines_per_burst_;
    int samples_per_burst_;
    int recommended_tile_area_ = 4000000;
    bool beam_dimap_mode_ = false;

    std::shared_ptr<AlusFileReader<double>> master_input_dataset_;
    std::shared_ptr<AlusFileReader<double>> slave_input_dataset_;
    std::shared_ptr<AlusFileWriter<float>> output_dataset_;
    std::shared_ptr<snapengine::Product> master_product_;
    std::shared_ptr<snapengine::Product> slave_product_;
    std::shared_ptr<snapengine::Product> target_product_;

    std::mutex register_mutex_;
    std::mutex master_read_mutex_;
    std::mutex position_compute_mutex_;
    std::mutex slave_read_mutex_;
    std::mutex core_compute_mutex_;
    std::mutex output_write_mutex_;
    std::mutex exception_mutex_;
    size_t worker_count_;         // How many were set loose
    size_t finished_count_;       // How many have finished work
    size_t active_worker_count_;  // How many are working concurrently
    std::condition_variable thread_sync_;
    std::condition_variable end_block_;

    std::string_view master_metadata_file_;
    std::string_view slave_metadata_file_;
    std::string swath_index_str_;
    std::string mst_suffix_;


    void CopySlaveMetadata(std::shared_ptr<snapengine::Product>& slaveProduct);
    void UpdateTargetProductMetadata();

    struct WorkerParams {
        int index;
        Rectangle master_input_area;
        Rectangle slave_input_area;
        size_t demod_size;
        int master_burst_index;
        int slave_burst_index;
    };

    class BackgeocodingWorker {
    public:
        BackgeocodingWorker(WorkerParams params, BackgeocodingController* controller) {
            params_ = params;
            controller_ = controller;
            std::thread worker(&BackgeocodingWorker::Work, this);
            worker.detach();
        }
        BackgeocodingWorker(const Backgeocoding&) = delete;
        BackgeocodingWorker& operator=(const Backgeocoding&) = delete;
        ~BackgeocodingWorker();
        void Work();

    private:
        WorkerParams params_;
        BackgeocodingController* controller_;
    };

    std::vector<BackgeocodingController::BackgeocodingWorker> workers_;
};

}  // namespace alus::backgeocoding
