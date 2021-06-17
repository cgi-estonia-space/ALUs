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
#include <map>
#include <memory>
#include <string_view>
#include <thread>

#include "alus_log.h"
#include "pointer_holders.h"
#include "snap-core/datamodel/metadata_element.h"
#include "snap-core/util/product_utils.h"
#include "snap-engine-utilities/datamodel/metadata/abstract_metadata.h"
#include "snap-engine-utilities/gpf/stack_utils.h"

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
      slave_metadata_file_(slave_metadata_file) {
    beam_dimap_mode_ = true;
}

BackgeocodingController::BackgeocodingController(std::shared_ptr<AlusFileReader<double>> master_input_dataset,
                                                 std::shared_ptr<AlusFileReader<double>> slave_input_dataset,
                                                 std::shared_ptr<AlusFileWriter<float>> output_dataset,
                                                 std::shared_ptr<snapengine::Product> master_product,
                                                 std::shared_ptr<snapengine::Product> slave_product)
    : master_input_dataset_(master_input_dataset),
      slave_input_dataset_(slave_input_dataset),
      output_dataset_(output_dataset),
      master_product_(master_product),
      slave_product_(slave_product) {}

BackgeocodingController::~BackgeocodingController() {}

void BackgeocodingController::PrepareToCompute(const float* egm96_device_array, PointerArray srtm3_tiles) {
    backgeocoding_ = std::make_unique<Backgeocoding>();
    backgeocoding_->SetElevationData(egm96_device_array, srtm3_tiles);
    if (beam_dimap_mode_) {
        backgeocoding_->PrepareToCompute(master_metadata_file_, slave_metadata_file_);
    } else {
        backgeocoding_->PrepareToCompute(master_product_, slave_product_);
    }

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
    // master_input_dataset_->ReadRectangle(master_area, 1, i_tile);
    // master_input_dataset_->ReadRectangle(master_area, 2, q_tile);
    std::map<int, double*> bands;
    bands.insert({1, i_tile});
    bands.insert({2, q_tile});
    master_input_dataset_->ReadRectangle(master_area, bands);
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
    // slave_input_dataset_->ReadRectangle(slave_area, 1, i_tile);
    // slave_input_dataset_->ReadRectangle(slave_area, 2, q_tile);
    std::map<int, double*> bands;
    bands.insert({1, i_tile});
    bands.insert({2, q_tile});
    slave_input_dataset_->ReadRectangle(slave_area, bands);
}

void BackgeocodingController::CoreCompute(CoreComputeParams params) {
    std::unique_lock<std::mutex> lock(core_compute_mutex_);

    backgeocoding_->CoreCompute(params);
}

void BackgeocodingController::WriteOutputs(Rectangle output_area, float* i_master_results, float* q_master_results,
                                           float* i_slave_results, float* q_slave_results) {
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
    LOGV << "Final block reached.";

    // make sure the last thread actually got out of its lock.
    register_mutex_.lock();
    LOGV << "Final thread release confirmed.";
    register_mutex_.unlock();
    if (exceptions_thrown_) {
        LOGW << "Backgeocoding had " << exceptions_thrown_ << " exceptions thrown. Passing on the first one.";
        std::rethrow_exception(exceptions_.at(0));
    }
}

void BackgeocodingController::Initialize() {
    if (master_product_ == nullptr || slave_product_ == nullptr) {
        LOGE << "master or slave product is null. Skipping backgeocoding initialization function.";
        return;
    }

    // TODO: do I need this?
    // checkSourceProductValidity();

    // TODO: implemented in a different way.
    /*mSU = new Sentinel1Utils(masterProduct);
    mSubSwath = mSU.getSubSwath();
    mSU.computeDopplerRate();
    mSU.computeReferenceTime();

    for(Product product : sourceProduct) {
        if(product.equals(masterProduct))
            continue;
        slaveDataList.add(new SlaveData(product));
    }*/

    /*
    outputToFile("c:\\output\\mSensorPosition.dat", mSU.getOrbit().sensorPosition);
    outputToFile("c:\\output\\mSensorVelocity.dat", mSU.getOrbit().sensorVelocity);
    outputToFile("c:\\output\\sSensorPosition.dat", sSU.getOrbit().sensorPosition);
    outputToFile("c:\\output\\sSensorVelocity.dat", sSU.getOrbit().sensorVelocity);
    */

    s1tbx::Sentinel1Utils* master_utils = backgeocoding_->GetMasterUtils();

    std::vector<std::string> mSubSwathNames = master_utils->GetSubSwathNames();
    std::vector<std::string> mPolarizations = master_utils->GetPolarizations();

    // TODO: not checking any of that. Is it needed?
    /*for(SlaveData slaveData : slaveDataList) {
        final String[] sSubSwathNames = slaveData.sSU.getSubSwathNames();
        if (mSubSwathNames.length != 1 || sSubSwathNames.length != 1) {
            throw new OperatorException("Split product is expected.");
        }

        if (!mSubSwathNames[0].equals(sSubSwathNames[0])) {
            throw new OperatorException("Same sub-swath is expected.");
        }

        final String[] sPolarizations = slaveData.sSU.getPolarizations();
        if (!StringUtils.containsIgnoreCase(sPolarizations, mPolarizations[0])) {
            throw new OperatorException("Same polarization is expected.");
        }
    }*/

    // subSwathIndex = 1; // subSwathIndex is always 1 because of split product
    swath_index_str_ = mSubSwathNames.at(0).substr(0, 3);

    // TODO: Currently only supported dem is srtm3
    // TODO: Currently only supports bilinear resampling
    /*if (externalDEMFile == null) {
        DEMFactory.checkIfDEMInstalled(demName);
    }

    DEMFactory.validateDEM(demName, masterProduct);

    selectedResampling = ResamplingFactory.createResampling(resamplingType);
    if(selectedResampling == null) {
        throw new OperatorException("Resampling method "+ resamplingType + " is invalid");
    }*/

    // TODO: implement this to create an output product.
    /*createTargetProduct();

    std::vector<std::string> masterProductBands;
    for (std::string bandName : master_product_->GetBandNames()) {
        if (master_product_->GetBand(bandName) instanceof VirtualBand) {
            continue;
        }
        masterProductBands.push_back(bandName + mst_suffix_);
    }

    StackUtils.saveMasterProductBandNames(targetProduct,
                                          masterProductBands.toArray(new String[masterProductBands.size()]));
    StackUtils.saveSlaveProductNames(sourceProduct, targetProduct,
                                     masterProduct, targetBandToSlaveBandMap);

    updateTargetProductMetadata();

    final Band masterBandI = getBand(masterProduct, "i_", swathIndexStr, mSU.getPolarizations()[0]);
    if(masterBandI != null && masterBandI.isNoDataValueUsed()) {
        noDataValue = masterBandI.getNoDataValue();
    }*/
}

void BackgeocodingController::CopySlaveMetadata(std::shared_ptr<snapengine::Product>& slaveProduct) {
    std::shared_ptr<snapengine::MetadataElement> target_slave_metadata_root =
        snapengine::AbstractMetadata::GetSlaveMetadata(target_product_->GetMetadataRoot());
    std::shared_ptr<snapengine::MetadataElement> slv_abs_metadata =
        snapengine::AbstractMetadata::GetAbstractedMetadata(slaveProduct);
    if (slv_abs_metadata != nullptr) {
        std::string time_stamp = snapengine::StackUtils::CreateBandTimeStamp(slaveProduct);
        LOGV << "backgeocoding made a timestamp" << time_stamp;
        std::shared_ptr<snapengine::MetadataElement> target_slave_metadata =
            std::make_shared<snapengine::MetadataElement>(slaveProduct->GetName() + time_stamp);
        target_slave_metadata_root->AddElement(target_slave_metadata);
        snapengine::ProductUtils::CopyMetadata(slv_abs_metadata, target_slave_metadata);
    }
}

/**
 * Update target product metadata.
 */
void BackgeocodingController::UpdateTargetProductMetadata() {
    std::shared_ptr<snapengine::MetadataElement> abs_tgt =
        snapengine::AbstractMetadata::GetAbstractedMetadata(target_product_);
    snapengine::AbstractMetadata::SetAttribute(abs_tgt, snapengine::AbstractMetadata::COREGISTERED_STACK, 1);

    // TODO: not entirely sure if needed
    /*std::shared_ptr<snapengine::MetadataElement> inputElem = ProductInformation::GetInputProducts(target_product_);
    for(SlaveData slaveData : slaveDataList) {
        std::shared_ptr<snapengine::MetadataElement> slvInputElem =
    ProductInformation.getInputProducts(slaveData.slaveProduct); final MetadataAttribute[] slvInputProductAttrbList =
    slvInputElem.getAttributes(); for (MetadataAttribute attrib : slvInputProductAttrbList) {
            std::shared_ptr<snapengine::MetadataElement> inputAttrb = AbstractMetadata.addAbstractedAttribute(
                inputElem, "InputProduct", ProductData.TYPE_ASCII, "", "");
            inputAttrb.getData().setElems(attrib.getData().getElemString());
        }
    }*/

    // TODO: not sure if anything gets hurt by this.
    // CreateStackOp.getBaselines(sourceProduct, target_product_);
}
}  // namespace alus::backgeocoding
