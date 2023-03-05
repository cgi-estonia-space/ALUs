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

#include "c16_dataset.h"

#include <chrono>
#include <map>
#include <memory>
#include <string_view>
#include <thread>

#include "alus_log.h"
#include "dem_property.h"
#include "dem_type.h"
#include "pointer_holders.h"
#include "snap-core/core/datamodel/metadata_element.h"
#include "snap-core/core/util/product_utils.h"
#include "snap-engine-utilities/engine-utilities/datamodel/metadata/abstract_metadata.h"
#include "snap-engine-utilities/engine-utilities/gpf/stack_utils.h"
#include "tile_queue.h"

#include "target_dataset.h"


void OSVLUTToConstantMem(const std::vector<double>& master, const std::vector<double>& slave);

namespace alus::backgeocoding {

BackgeocodingController::BackgeocodingController(std::shared_ptr<AlusFileReader<int16_t>> master_input_dataset,
                                                 std::shared_ptr<AlusFileReader<int16_t>> slave_input_dataset,
                                                 std::shared_ptr<TargetDataset<float>> output_dataset,
                                                 std::string_view master_metadata_file,
                                                 std::string_view slave_metadata_file)
    : beam_dimap_mode_(true),
      master_input_dataset_(std::move(master_input_dataset)),
      slave_input_dataset_(std::move(slave_input_dataset)),
      output_dataset_(std::move(output_dataset)),
      master_metadata_file_(master_metadata_file),
      slave_metadata_file_(slave_metadata_file) {}

BackgeocodingController::BackgeocodingController(std::shared_ptr<AlusFileReader<int16_t>> master_input_dataset,
                                                 std::shared_ptr<AlusFileReader<int16_t>> slave_input_dataset,
                                                 std::shared_ptr<TargetDataset<float>> output_dataset,
                                                 std::shared_ptr<snapengine::Product> master_product,
                                                 std::shared_ptr<snapengine::Product> slave_product)
    : master_input_dataset_(std::move(master_input_dataset)),
      slave_input_dataset_(std::move(slave_input_dataset)),
      output_dataset_(std::move(output_dataset)),
      master_product_(std::move(master_product)),
      slave_product_(std::move(slave_product)) {}

void BackgeocodingController::PrepareToCompute(const float* egm96_device_array, PointerArray dem_tiles,
                                               bool mask_out_area_without_elevation,
                                               const dem::Property* device_dem_properties,
                                               const std::vector<dem::Property>& dem_properties, dem::Type dem_type) {
    backgeocoding_ = std::make_unique<Backgeocoding>();
    backgeocoding_->SetElevationData(egm96_device_array, dem_tiles, mask_out_area_without_elevation,
                                     device_dem_properties, dem_properties, dem_type);
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
    std::unique_lock lock(exception_mutex_);

    exceptions_.push_back(std::move(e));
    exceptions_thrown_++;
}

void BackgeocodingController::ReadMaster(Rectangle master_area, int16_t* i_tile, int16_t* q_tile) const {
    // find out if the band ordering is random or not. Then replace those numbers?
    std::map<int, int16_t*> bands;
    bands.insert({1, i_tile});
    bands.insert({2, q_tile});
    master_input_dataset_->ReadRectangle(master_area, bands);
}

PositionComputeResults BackgeocodingController::PositionCompute(int m_burst_index, int s_burst_index,
                                                                Rectangle target_area, double* device_x_points,
                                                                double* device_y_points, ComputeCtx* ctx) {
    PositionComputeResults result;
    result.slave_area =
        backgeocoding_->PositionCompute(m_burst_index, s_burst_index, target_area, device_x_points, device_y_points, ctx);
    result.demod_size = result.slave_area.width * result.slave_area.height;

    return result;
}

void BackgeocodingController::ReadSlave(Rectangle slave_area, int16_t* i_tile, int16_t* q_tile) const {
    // find out if the band ordering is random or not. Then replace those numbers?
    std::map<int, int16_t*> bands;
    bands.insert({1, i_tile});
    bands.insert({2, q_tile});
    slave_input_dataset_->ReadRectangle(slave_area, bands);
}

void BackgeocodingController::CoreCompute(const CoreComputeParams& params) const {
    backgeocoding_->CoreCompute(params);
}

void BackgeocodingController::WriteOutputs(Rectangle output_area, float* /*i_master_results*/, float* /*q_master_results*/,
                                           float* i_slave_results, float* q_slave_results) const {
    //output_dataset_->WriteRectangle(i_master_results, output_area, 1);
    //output_dataset_->WriteRectangle(q_master_results, output_area, 2);
    output_dataset_->WriteRectangle(i_slave_results, output_area, 3);
    output_dataset_->WriteRectangle(q_slave_results, output_area, 4);
}


inline std::vector<double> CalculateOrbitStateVectorLUT(
    const std::vector<alus::snapengine::OrbitStateVectorComputation>& comp_orbits) {
    const auto& osv = comp_orbits;
    std::vector<double> h_lut;
    for (size_t i = 0; i < osv.size(); i++) {
        for (size_t j = 0; j < osv.size(); j++) {
            double timei = osv[i].timeMjd_;
            double timej = osv[j].timeMjd_;
            if (timei != timej) {
                h_lut.push_back(1 / (timei - timej));
            } else {
                h_lut.push_back(0);
            }
        }
    }
    return h_lut;
}

void IQ16To2xFloatGdal(GDALDataset* in, GDALDataset* i_out, GDALDataset* q_out)
{
    auto b_in = in->GetRasterBand(1);
    auto b_i = i_out->GetRasterBand(1);
    auto b_q = q_out->GetRasterBand(1);

    int y_size = b_in->GetYSize();
    int x_size = b_in->GetXSize();

    if(y_size != b_i->GetYSize() || y_size != b_q->GetYSize())
    {
        throw std::runtime_error("Experimental full swath code only!");
    }


    std::vector<Iq16> iq_in(x_size);
    std::vector<float> i_conv(x_size);
    std::vector<float> q_conv(x_size);
    for(int i = 0; i < y_size; i++)
    {
        CHECK_GDAL_ERROR(b_in->ReadBlock(0, i, iq_in.data()));
        for(int j = 0; j < x_size; j++)
        {
            i_conv[j] = iq_in[j].i;
            q_conv[j] = iq_in[j].q;
        }
        CHECK_GDAL_ERROR(b_i->WriteBlock(0, i, i_conv.data()));
        CHECK_GDAL_ERROR(b_q->WriteBlock(0, i, q_conv.data()));
    }
}

void BackgeocodingController::DoWork() {
    WorkerParams params;
    exceptions_thrown_ = 0;
    const int slave_burst_offset = backgeocoding_->GetBurstOffset();

    std::vector<BackgeocodingController::BackgeocodingWorker> workers;
    const int recommended_width = recommended_tile_area_ / lines_per_burst_;
    int worker_count = 0;
    params.index = 0;

    auto b = std::chrono::steady_clock::now();
    auto master_osv = backgeocoding_->GetMasterUtils()->GetOrbitStateVectors()->orbit_state_vectors_computation_;
    auto slave_osv = backgeocoding_->GetMasterUtils()->GetOrbitStateVectors()->orbit_state_vectors_computation_;;
    auto master_lut = CalculateOrbitStateVectorLUT(master_osv);
    auto slave_lut = CalculateOrbitStateVectorLUT(slave_osv);

    OSVLUTToConstantMem(master_lut, slave_lut);
    auto e = std::chrono::steady_clock::now();

    auto diff = std::chrono::duration_cast<std::chrono::microseconds>(e-b).count();

    LOGI << "OSV time = " << diff << " ms";

    LOGI << "OSV sizes = " << master_lut.size() << " " << slave_lut.size();




    for (int burst_index = 0; burst_index < num_of_bursts_; burst_index++) {
        const int first_line_idx = burst_index * lines_per_burst_;

        for (int sample_index = 0; sample_index < samples_per_burst_; sample_index += recommended_width) {
            const int actual_width = (sample_index + recommended_width < samples_per_burst_)
                                         ? recommended_width
                                         : samples_per_burst_ - sample_index;
            params.index++;
            params.master_input_area = {sample_index, first_line_idx, actual_width, lines_per_burst_};
            params.slave_burst_index = burst_index + slave_burst_offset;
            params.master_burst_index = burst_index;
            worker_count++;
            workers.emplace_back(params, this);
        }
    }


    std::shared_ptr<C16Dataset<int16_t>> master_rd = std::dynamic_pointer_cast<C16Dataset<int16_t>>(master_input_dataset_);

    auto master_gdal = master_rd->GetDataset()->GetGdalDataset();

    GDALDataset* i_master = output_dataset_->GetDataset()[0];
    GDALDataset* q_master = output_dataset_->GetDataset()[1];


    std::thread master_read_thread(IQ16To2xFloatGdal, master_gdal, i_master, q_master);

    //master_read_thread.join();

    ThreadSafeTileQueue<BackgeocodingWorker> queue(std::move(workers));
    std::vector<std::thread> threads_vec;

    // Backgeocoding does a lot of things on cpu, 2 x input datasets, 4 x output datasets, partial CPU triangulation
    // this number should be double checked if further optimizations are made
    const int n_worker_threads = 10;
    LOGI << "n_worker_threads = " << n_worker_threads;
    threads_vec.reserve(n_worker_threads);

    std::vector<ComputeCtx> ctx_vec(n_worker_threads);

    for(auto& ctx : ctx_vec)
    {
        cudaStreamCreate(&ctx.stream);
        //LOGI << "stream = " << ctx.stream;
    }


    for (int i = 0; i < n_worker_threads; i++) {
        ComputeCtx* ctx = &ctx_vec[i];
        threads_vec.emplace_back([&queue, ctx]() {
            BackgeocodingWorker worker;
            while (queue.PopFront(worker)) {
                worker.Work(ctx);
            }
        });
    }

    for (auto& thread : threads_vec) {
        thread.join();
    }

    master_read_thread.join();


    for(auto& ctx : ctx_vec)
    {
        //LOGI<< "destroy " << ctx.stream;
        cudaStreamDestroy(ctx.stream);
    }

    LOGV << "Final block reached.";
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

    // do I need this?
    // checkSourceProductValidity();

    // for now, implemented in a different way.
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

    std::vector<std::string> m_sub_swath_names = master_utils->GetSubSwathNames();
    // std::vector<std::string> m_polarizations = master_utils->GetPolarizations();

    // TODO(unknown): not checking any of that. Is it needed?
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
    swath_index_str_ = m_sub_swath_names.at(0).substr(0, 3);

    // Currently only supported dem is srtm3
    // Currently only supports bilinear resampling
    /*if (externalDEMFile == null) {
        DEMFactory.checkIfDEMInstalled(demName);
    }

    DEMFactory.validateDEM(demName, masterProduct);

    selectedResampling = ResamplingFactory.createResampling(resamplingType);
    if(selectedResampling == null) {
        throw new OperatorException("Resampling method "+ resamplingType + " is invalid");
    }*/

    // TODO(unknown): implement this to create an output product.
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

    // TODO(unknown): not entirely sure if needed
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

    // TODO(unknown): not sure if anything gets hurt by this.
    // CreateStackOp.getBaselines(sourceProduct, target_product_);
}
}  // namespace alus::backgeocoding
