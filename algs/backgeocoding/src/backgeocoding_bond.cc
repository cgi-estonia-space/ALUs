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
#include "backgeocoding_controller.h"
#include "dataset.h"
#include "pointer_holders.h"
#include "raster_properties.hpp"
#include "target_dataset.h"

namespace alus::backgeocoding {

void BackgeocodingBond::SetParameters(const app::AlgorithmParameters::Table& param_values) {
    int input_count = 0;

    auto master_input_dataset = param_values.find(std::string(PARAMETER_MASTER_PATH));
    if (master_input_dataset != param_values.end()) {
        master_input_dataset_ = std::make_shared<Dataset<double>>(master_input_dataset->second);
        input_count++;
    }

    auto slave_input_dataset = param_values.find(std::string(PARAMETER_SLAVE_PATH));
    if (slave_input_dataset != param_values.end()) {
        slave_input_dataset_ = std::make_shared<Dataset<double>>(slave_input_dataset->second);
        input_count++;
    }

    auto output_path = param_values.find(std::string(PARAMETER_OUTPUT_PATH));
    if (output_path != param_values.end()) {
        alus::TargetDatasetParams params;
        params.filename = output_path->second;
        params.band_count = 4;
        params.driver = master_input_dataset_->GetGdalDataset()->GetDriver();
        params.dimension = master_input_dataset_->GetRasterDimensions();
        params.transform = master_input_dataset_->GetTransform();
        params.projectionRef = master_input_dataset_->GetGdalDataset()->GetProjectionRef();

        output_dataset_ = std::make_shared<TargetDataset<float>>(params);
        outputs_set_ = true;
    }

    auto master_metadata_path = param_values.find(std::string(PARAMETER_MASTER_METADATA));
    if (master_metadata_path != param_values.end()) {
        master_metadata_path_ = master_metadata_path->second;
        input_count++;
    }

    auto slave_metadata_path = param_values.find(std::string(PARAMETER_SLAVE_METADATA));
    if (slave_metadata_path != param_values.end()) {
        slave_metadata_path_ = slave_metadata_path->second;
        input_count++;
    }
    inputs_set_ = input_count == 4;
}

void BackgeocodingBond::SetSrtm3Buffers(const PointerHolder* buffers, size_t length) {
    srtm3_tiles_ = {const_cast<PointerHolder*>(buffers), length};
    srtm3_set_ = true;
}

void BackgeocodingBond::SetEgm96Buffers(const float* egm96_device_array) {
    egm96_device_array_ = egm96_device_array;
    egm96_set_ = true;
}

void BackgeocodingBond::SetTileSize(size_t /*width*/, size_t /*height*/) {
    std::cout << "Backgeocoding does not support custom tile sizes at the moment. Ignoring this" << std::endl;
}

void BackgeocodingBond::SetOutputFilename([[maybe_unused]] const std::string& /*output_name*/) {
    std::cout << "Backgeocoding is ignoring the SetOutputFilename function." << std::endl;
}

int BackgeocodingBond::Execute() {
    if (inputs_set_ && outputs_set_ && srtm3_set_ && egm96_set_) {
        controller_ = std::make_unique<BackgeocodingController>(
            master_input_dataset_, slave_input_dataset_, output_dataset_, master_metadata_path_, slave_metadata_path_);
        controller_->PrepareToCompute(egm96_device_array_, srtm3_tiles_);
        controller_->DoWork();
    } else {
        std::string message = inputs_set_    ? "Incorrect amount of backgeocoding inputs set."
                              : outputs_set_ ? "No backgeocoding outputs specified."
                              : srtm3_set_   ? "Srtm3 tiles not set for backgeocoding."
                              : egm96_set_   ? "Egm96 not set for backgeocding."
                                             : "Backgeocoding bond arrived at an impossible error.";
        throw std::runtime_error(message);
    }

    return 0;
}

void BackgeocodingBond::SetInputFilenames([[maybe_unused]] const std::string& input_dataset,
                                          [[maybe_unused]] const std::string& metadata_path) {
    std::cout << "Backgeocoding will be ignoring the SetInputs function." << std::endl;
}

[[nodiscard]] std::string BackgeocodingBond::GetArgumentsHelp() const {
    boost::program_options::options_description options("Backgeocoding argument list");
    options.add_options()(std::string(PARAMETER_MASTER_PATH).c_str(), "Master tif file.")(
        std::string(PARAMETER_SLAVE_PATH).c_str(), "Slave tif file")(std::string(PARAMETER_MASTER_METADATA).c_str(),
                                                                     "Master metadata file(.dim)")(
        std::string(PARAMETER_SLAVE_METADATA).c_str(), "Slave metadata file(.dim)")(
        std::string(PARAMETER_OUTPUT_PATH).c_str(), "End product file name.")(
        std::string(PARAMETER_MASK_ELEVATION).c_str(), " Use of elevation mask 1 or 0. (Currently not implemented)");

    std::stringstream result;
    result << options;
    return boost::algorithm::replace_all_copy(result.str(), "--", "");
}

BackgeocodingBond::~BackgeocodingBond() = default;

}  // namespace alus::backgeocoding

extern "C" {
alus::AlgBond* CreateAlgorithm() { return new alus::backgeocoding::BackgeocodingBond(); }

void DeleteAlgorithm(alus::AlgBond* instance) { delete (alus::backgeocoding::BackgeocodingBond*)instance; }
}