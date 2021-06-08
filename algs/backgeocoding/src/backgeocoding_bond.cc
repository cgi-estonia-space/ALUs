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
#include "backgeocoding_bond.h"

#include <memory>
#include <sstream>
#include <string>
#include <string_view>

#include <boost/algorithm/string.hpp>
#include <boost/program_options.hpp>

#include "algorithm_parameters.h"
#include "alus_log.h"
#include "backgeocoding_controller.h"
#include "dataset.h"
#include "earth_gravitational_model96.h"
#include "srtm3_elevation_model.h"
#include "target_dataset.h"

namespace {
constexpr std::string_view PARAMETER_MASTER_PATH{"master"};
constexpr std::string_view PARAMETER_MASTER_METADATA{"master_metadata"};
constexpr std::string_view PARAMETER_MASK_ELEVATION{"mask_elevation"};  // TODO: this will be used some day
}  // namespace

namespace alus::backgeocoding {

void BackgeocodingBond::SetParameters(const app::AlgorithmParameters::Table& param_values) {
    if (auto master_input_dataset = param_values.find(std::string(PARAMETER_MASTER_PATH));
        master_input_dataset != param_values.end()) {
        master_dataset_arg_ = master_input_dataset->second;
    }

    if (auto master_metadata_path = param_values.find(std::string(PARAMETER_MASTER_METADATA));
        master_metadata_path != param_values.end()) {
        master_metadata_arg_ = master_metadata_path->second;
    }
}

void BackgeocodingBond::SetSrtm3Manager(snapengine::Srtm3ElevationModel* manager) { srtm3_manager_ = manager; }

void BackgeocodingBond::SetEgm96Manager(const snapengine::EarthGravitationalModel96* manager) {
    egm96_manager_ = manager;
}

void BackgeocodingBond::SetTileSize(size_t /*width*/, size_t /*height*/) {
    LOGI << "Backgeocoding does not support custom tile sizes at the moment. Ignoring this";
}

void BackgeocodingBond::SetOutputFilename([[maybe_unused]] const std::string& output_name) {
    output_filename_ = output_name;
}

int BackgeocodingBond::Execute() {
    try {
        if (input_dataset_filenames_.size() != 2 || input_metadata_filenames_.size() != 2) {
            LOGE << "Backgeocoding requires exactly 2 input datasets and 2 metadata files.";
            return 3;
        }

        if (srtm3_manager_ == nullptr) {
            LOGE << "SRTM3 manager not set for backgeocoding.";
            return 3;
        }

        if (egm96_manager_ == nullptr) {
            LOGE << "EGM96 manager not set for backgeocoding.";
            return 3;
        }

        if (master_dataset_arg_.empty()) {
            LOGE << "Master dataset argument is empty.";
            return 3;
        }

        std::string master_input_dataset_filename{};
        std::string slave_input_dataset_filename{};
        for (const auto& ds_name : input_dataset_filenames_) {
            if (ds_name.find(master_dataset_arg_, 0) != std::string::npos) {
                master_input_dataset_filename = ds_name;
            } else {
                slave_input_dataset_filename = ds_name;
            }
        }

        if (master_input_dataset_filename.empty()) {
            LOGE << "Cannot determine master dataset path (" << master_dataset_arg_ << " is missing in inputs).";
            return 3;
        }

        if (master_metadata_arg_.empty()) {
            LOGE << "Master metadata argument is empty.";
            return 3;
        }

        std::string master_input_metadata_filename{};
        std::string slave_input_metadata_filename{};
        for (const auto& dim_name : input_metadata_filenames_) {
            if (dim_name.find(master_metadata_arg_, 0) != std::string::npos) {
                master_input_metadata_filename = dim_name;
            } else {
                slave_input_metadata_filename = dim_name;
            }
        }

        if (master_input_metadata_filename.empty()) {
            LOGE << "Cannot determine master metadata path (" << master_metadata_arg_ << " is missing in inputs).";
            return 3;
        }

        auto master_input_dataset = std::make_shared<Dataset<double>>(master_input_dataset_filename);
        auto slave_input_dataset = std::make_shared<Dataset<double>>(slave_input_dataset_filename);
        alus::TargetDatasetParams params;
        params.filename = output_filename_;
        params.band_count = 4;
        params.driver = master_input_dataset->GetGdalDataset()->GetDriver();
        params.dimension = master_input_dataset->GetRasterDimensions();
        params.transform = master_input_dataset->GetTransform();
        params.projectionRef = master_input_dataset->GetGdalDataset()->GetProjectionRef();

        auto output_dataset = std::make_shared<TargetDataset<float>>(params);

        srtm3_manager_->HostToDevice();
        auto controller =
            std::make_unique<BackgeocodingController>(master_input_dataset, slave_input_dataset, output_dataset,
                                                      master_input_metadata_filename, slave_input_metadata_filename);
        controller->PrepareToCompute(egm96_manager_->GetDeviceValues(), {srtm3_manager_->GetSrtmBuffersInfo(),
                                                                         srtm3_manager_->GetDeviceSrtm3TilesCount()});
        controller->DoWork();

        srtm3_manager_->DeviceFree();
    } catch (const std::exception& e) {
        LOGE << "Exception caught while running Backgeocoding - " << e.what();
        return 1;
    } catch (...) {
        LOGE << "Unknown exception caught while running Backgeocoding";
        return 2;
    }

    return 0;
}

void BackgeocodingBond::SetInputFilenames([[maybe_unused]] const std::vector<std::string>& input_datasets,
                                          [[maybe_unused]] const std::vector<std::string>& metadata_paths) {
    input_dataset_filenames_ = input_datasets;
    input_metadata_filenames_ = metadata_paths;
}

[[nodiscard]] std::string BackgeocodingBond::GetArgumentsHelp() const {
    boost::program_options::options_description options("Backgeocoding argument list");
    options.add_options()(std::string(PARAMETER_MASTER_PATH).c_str(), "Master dataset filename/dataset ID.")(
        std::string(PARAMETER_MASTER_METADATA).c_str(),
        "Master metadata file ID (.dim)")(std::string(PARAMETER_MASK_ELEVATION).c_str(),
                                          "Use of elevation mask 1 or 0. (Currently not implemented)");

    std::stringstream result;
    result << options;
    return boost::algorithm::replace_all_copy(result.str(), "--", "");
}

}  // namespace alus::backgeocoding