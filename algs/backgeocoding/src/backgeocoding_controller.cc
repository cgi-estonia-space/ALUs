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

#include <chrono>
#include <memory>
#include <string_view>
#include <thread>

#include "pointer_holders.h"

namespace alus::backgeocoding {

BackgeocodingController::BackgeocodingController(std::shared_ptr<AlusFileReader<double>> master_input_dataset,
                                                 std::shared_ptr<AlusFileReader<double>> slave_input_dataset,
                                                 std::shared_ptr<AlusFileWriter<float>> output_dataset,
                                                 std::string_view master_metadata_file,
                                                 std::string_view slave_metadata_file)
    : master_input_dataset_(master_input_dataset),
      slave_input_dataset_(slave_input_dataset),
      output_dataset_(output_dataset),
      master_metadata_file_(master_metadata_file),
      slave_metadata_file_(slave_metadata_file) {}

BackgeocodingController::~BackgeocodingController() {}

void BackgeocodingController::PrepareToCompute(const float* egm96_device_array, PointerArray srtm3_tiles) {
    backgeocoding_ = std::make_unique<Backgeocoding>();
    backgeocoding_->SetElevationData(egm96_device_array, srtm3_tiles);
    backgeocoding_->PrepareToCompute(master_metadata_file_, slave_metadata_file_);

    num_of_bursts_ = backgeocoding_->GetNrOfBursts();
    lines_per_burst_ = backgeocoding_->GetLinesPerBurst();
    samples_per_burst_ = backgeocoding_->GetSamplesPerBurst();
}

void BackgeocodingController::RegisterException(std::exception_ptr e) {
    std::unique_lock<std::mutex> lock(exception_mutex_);

    exceptions_.push_back(e);
    exceptions_thrown_++;
}

void BackgeocodingController::RegisterThreadEnd() {
    std::unique_lock<std::mutex> lock(register_mutex_);

    finished_count_++;

    if (finished_count_ == worker_count_) {
        end_block_.notify_all();
    } else {
        thread_sync_.notify_one();
    }
}

void BackgeocodingController::ReadMaster(Rectangle master_area, double* i_tile, double* q_tile) {
    std::unique_lock<std::mutex> lock(master_read_mutex_);
    // TODO: Could we read slave and master at the same time if we branch into 2 other threads during this thread.
    // TODO: find out if the band ordering is random or not. Then replace those numbers.
    master_input_dataset_->ReadRectangle(master_area, 1, i_tile);
    master_input_dataset_->ReadRectangle(master_area, 2, q_tile);
}

PositionComputeResults BackgeocodingController::PositionCompute(int m_burst_index, int s_burst_index,
                                                                Rectangle target_area, double* device_x_points,
                                                                double* device_y_points) {
    PositionComputeResults result;
    std::unique_lock<std::mutex> lock(position_compute_mutex_);

    result.slave_area =
        backgeocoding_->PositionCompute(m_burst_index, s_burst_index, target_area, device_x_points, device_y_points);
    result.demod_size = result.slave_area.width * result.slave_area.height;

    return result;
}

void BackgeocodingController::ReadSlave(Rectangle slave_area, double* i_tile, double* q_tile) {
    std::unique_lock<std::mutex> lock(slave_read_mutex_);

    // TODO: find out if the band ordering is random or not. Then replace those numbers.
    slave_input_dataset_->ReadRectangle(slave_area, 1, i_tile);
    slave_input_dataset_->ReadRectangle(slave_area, 2, q_tile);
}

void BackgeocodingController::CoreCompute(CoreComputeParams params) {
    std::unique_lock<std::mutex> lock(core_compute_mutex_);

    backgeocoding_->CoreCompute(params);
}

void BackgeocodingController::WriteOutputs(Rectangle output_area, float* i_master_results, float* q_master_results, float* i_slave_results, float* q_slave_results) {
    std::unique_lock<std::mutex> lock(output_write_mutex_);

    output_dataset_->WriteRectangle(i_master_results, output_area, 1);
    output_dataset_->WriteRectangle(q_master_results, output_area, 2);

    output_dataset_->WriteRectangle(i_slave_results, output_area, 3);
    output_dataset_->WriteRectangle(q_slave_results, output_area, 4);
}

void BackgeocodingController::DoWork() {
    WorkerParams params;
    exceptions_thrown_ = 0;
    worker_count_ = 1;
    finished_count_ = 0;
    active_worker_count_ = 5;
    int slave_burst_offset = backgeocoding_->GetBurstOffset();
    int first_line_idx;
    int recommended_width;
    int actual_width;

    recommended_width = recommended_tile_area_ / lines_per_burst_;
    worker_count_ = num_of_bursts_ * (lines_per_burst_ * samples_per_burst_ / recommended_tile_area_ + 1);
    workers_.reserve(worker_count_);
    worker_count_ = 0;
    worker_counter_ = 0;
    params.index = 0;

    for (int burst_index = 0; burst_index < num_of_bursts_; burst_index++) {
        first_line_idx = burst_index * lines_per_burst_;

        for (int sample_index = 0; sample_index < samples_per_burst_; sample_index += recommended_width) {
            actual_width = (sample_index + recommended_width < samples_per_burst_) ? recommended_width
                                                                                   : samples_per_burst_ - sample_index;
            params.index++;
            params.master_input_area = {sample_index, first_line_idx, actual_width, lines_per_burst_};
            params.slave_burst_index = burst_index + slave_burst_offset;
            params.master_burst_index = burst_index;
            worker_count_++;
            workers_.emplace_back(params, this);
        }
    }

    while (worker_counter_ < worker_count_) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    for (size_t i = 0; i < active_worker_count_ && i < worker_count_; i++) {
        thread_sync_.notify_one();
    }

    std::mutex final_mutex;
    std::unique_lock<std::mutex> final_lock(final_mutex);
    end_block_.wait(final_lock);
    std::cout << "Final block reached." << std::endl;

    // make sure the last thread actually got out of its lock.
    register_mutex_.lock();
    std::cout << "Final thread release confirmed." << std::endl;
    register_mutex_.unlock();
    if (exceptions_thrown_) {
        std::cout << "Backgeocoding had " << exceptions_thrown_ << " exceptions thrown. Passing on the first one."
                  << std::endl;
        std::rethrow_exception(exceptions_.at(0));
    }
}

}  // namespace alus::backgeocoding
