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
#include "dem_property.h"
#include "dem_type.h"
#include "pointer_holders.h"
#include "snap-core/core/datamodel/product.h"

#include "target_dataset.h"

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
    const float output_no_data_value_ = 0.0;
    int exceptions_thrown_ = 0;

    BackgeocodingController(std::shared_ptr<AlusFileReader<int16_t>> master_input_dataset,
                            std::shared_ptr<AlusFileReader<int16_t>> slave_input_dataset,
                            std::shared_ptr<TargetDataset<float>> output_dataset,
                            std::string_view master_metadata_file, std::string_view slave_metadata_file);

    BackgeocodingController(std::shared_ptr<AlusFileReader<int16_t>> master_input_dataset,
                            std::shared_ptr<AlusFileReader<int16_t>> slave_input_dataset,
                            std::shared_ptr<TargetDataset<float>> output_dataset,
                            std::shared_ptr<snapengine::Product> master_product,
                            std::shared_ptr<snapengine::Product> slave_product);

    ~BackgeocodingController() = default;
    BackgeocodingController(const BackgeocodingController&) = delete;  // class does not support copying(and moving)
    BackgeocodingController& operator=(const BackgeocodingController&) = delete;

    void PrepareToCompute(const float* egm96_device_array, PointerArray dem_tiles,
                          bool mask_out_area_without_elevation, const dem::Property* device_dem_properties,
                          const std::vector<dem::Property>& dem_properties, dem::Type dem_type);
    void RegisterException(std::exception_ptr e);
    void ReadMaster(Rectangle master_area, int16_t* i_tile, int16_t* q_tile) const;
    PositionComputeResults PositionCompute(int m_burst_index, int s_burst_index, Rectangle target_area,
                                           double* device_x_points, double* device_y_points, ComputeCtx* ctx);
    void ReadSlave(Rectangle slave_area, int16_t* i_tile, int16_t* q_tile) const;
    void CoreCompute(const CoreComputeParams& params) const;
    void WriteOutputs(Rectangle output_area, float* i_master_results, float* q_master_results, float* i_slave_results,
                      float* q_slave_results) const;
    void DoWork();
    void Initialize();

private:
    std::unique_ptr<Backgeocoding> backgeocoding_;
    std::vector<std::exception_ptr> exceptions_;
    int num_of_bursts_;
    int lines_per_burst_;
    int samples_per_burst_;
    const int recommended_tile_area_ = 4000000;
    bool beam_dimap_mode_ = false;

    std::shared_ptr<AlusFileReader<int16_t>> master_input_dataset_;
    std::shared_ptr<AlusFileReader<int16_t>> slave_input_dataset_;
    std::shared_ptr<TargetDataset<float>> output_dataset_;
    std::shared_ptr<snapengine::Product> master_product_;
    std::shared_ptr<snapengine::Product> slave_product_;
    std::shared_ptr<snapengine::Product> target_product_;

    std::mutex exception_mutex_;

    std::string_view master_metadata_file_;
    std::string_view slave_metadata_file_;
    std::string swath_index_str_;
    std::string mst_suffix_;

    void CopySlaveMetadata(std::shared_ptr<snapengine::Product>& slaveProduct);
    void UpdateTargetProductMetadata();

    struct WorkerParams {
        int index;
        Rectangle master_input_area;
        int master_burst_index;
        int slave_burst_index;
    };

    class BackgeocodingWorker {
    public:
        BackgeocodingWorker() = default;
        BackgeocodingWorker(WorkerParams params, BackgeocodingController* controller) {
            params_ = params;
            controller_ = controller;
        }
        BackgeocodingWorker(const Backgeocoding&) = delete;
        BackgeocodingWorker& operator=(const Backgeocoding&) = delete;
        void Work(ComputeCtx* ctx);

    private:
        WorkerParams params_;
        BackgeocodingController* controller_;
    };
};

}  // namespace alus::backgeocoding
