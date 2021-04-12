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

#include "coherence_estimation_routine_execute.h"

#include <dlfcn.h>
#include <iostream>
#include <map>
#include <sstream>
#include <string_view>

#include <gdal.h>

#include "alg_load.h"
#include "gdal_util.h"

namespace {
constexpr std::string_view EXECUTOR_NAME{"Coherence estimation routine"};
constexpr std::string_view PARAMETER_ID_APPLY_ORBIT_FILE_OP_LIBRARY{"apply_orbit_file_lib"};
constexpr std::string_view PARAMETER_ID_BACKGEOCODING_OP_LIBRARY{"backgeocoding_lib"};
constexpr std::string_view PARAMETER_ID_COHERENCE_OP_LIBRARY{"coherence_lib"};
constexpr std::string_view PARAMETER_ID_DEBURST_OP_LIBRARY{"deburst_lib"};
constexpr std::string_view PARAMETER_ID_RANGE_DOPPLER_TERRAIN_CORRECTION_LIBRARY{"terrain_correction_lib"};
}  // namespace

namespace alus {

int CoherenceEstimationRoutineExecute::Execute() {
    PrintProcessingParameters();

    const std::vector<std::string> routine_algs{coherence_lib_, range_doppler_terrain_correction_lib_};

    GDALAllRegister();
    auto rolling_input_dataset = static_cast<GDALDataset*>(GDALOpen(input_dataset_.data(), GA_ReadOnly));
    CHECK_GDAL_PTR(rolling_input_dataset);

    auto const po_driver = GetGDALDriverManager()->GetDriverByName("MEM");
    CHECK_GDAL_PTR(po_driver);

    for (const auto& alg_lib_name : routine_algs) {
        const AlgorithmLoadGuard alg_guard{alg_lib_name};
        alg_guard.GetInstanceHandle()->SetParameters(alg_params_);
        alg_guard.GetInstanceHandle()->SetSrtm3Buffers(srtm3_buffers_, srtm3_buffers_length_);
        alg_guard.GetInstanceHandle()->SetTileSize(tile_width_, tile_height_);
        alg_guard.GetInstanceHandle()->SetInputDataset(rolling_input_dataset, metadata_path_);

        if (alg_lib_name == routine_algs.back()) {
            alg_guard.GetInstanceHandle()->SetOutputFilename(output_name_);
        } else {
            alg_guard.GetInstanceHandle()->SetOutputDriver(po_driver);
        }

        try {
            const auto res = alg_guard.GetInstanceHandle()->Execute();

            if (alg_lib_name != routine_algs.back()) {
                GDALClose(rolling_input_dataset);
                rolling_input_dataset = nullptr;
                rolling_input_dataset = alg_guard.GetInstanceHandle()->GetProcessedDataset();
            }

            if (res != 0) {
                std::cout << "Running " << alg_lib_name << " resulted in non success execution - " << res << std::endl
                          << "Aborting." << std::endl;
                return res;
            }
        } catch (const std::exception& e) {
            std::cout << "Running " << alg_lib_name << " resulted in error:" << e.what() << std::endl
                      << "Aborting." << std::endl;
            if (rolling_input_dataset != nullptr) {
                GDALClose(rolling_input_dataset);
            }
            return 2;
        }
    }

    return 0;
}

void CoherenceEstimationRoutineExecute::SetInputFilenames(const std::string& input_dataset,
                                                          const std::string& metadata_path) {
    input_dataset_ = input_dataset;
    metadata_path_ = metadata_path;
}

void CoherenceEstimationRoutineExecute::SetParameters(const app::AlgorithmParameters::Table& param_values) {
    const std::map<std::string_view, std::string&> string_parameters_var_table{
        {PARAMETER_ID_APPLY_ORBIT_FILE_OP_LIBRARY, apply_orbit_file_lib_},
        {PARAMETER_ID_BACKGEOCODING_OP_LIBRARY, backgeocoding_lib_},
        {PARAMETER_ID_COHERENCE_OP_LIBRARY, coherence_lib_},
        {PARAMETER_ID_DEBURST_OP_LIBRARY, deburst_lib_},
        {PARAMETER_ID_RANGE_DOPPLER_TERRAIN_CORRECTION_LIBRARY, range_doppler_terrain_correction_lib_}};

    for (const auto& parameter : string_parameters_var_table) {
        auto value_it = param_values.find(std::string(parameter.first));
        if (value_it != param_values.end()) {
            parameter.second = value_it->second;
        }
    }

    alg_params_ = param_values;
}

void CoherenceEstimationRoutineExecute::SetSrtm3Buffers(const PointerHolder* buffers, size_t length) {
    srtm3_buffers_ = buffers;
    srtm3_buffers_length_ = length;
}

void CoherenceEstimationRoutineExecute::SetTileSize(size_t width, size_t height) {
    tile_width_ = width;
    tile_height_ = height;
}

void CoherenceEstimationRoutineExecute::SetOutputFilename(const std::string& output_name) {
    output_name_ = output_name;
}

std::string CoherenceEstimationRoutineExecute::GetArgumentsHelp() const {
    std::stringstream help_stream;
    help_stream << EXECUTOR_NAME << " parameters:" << std::endl
                << PARAMETER_ID_APPLY_ORBIT_FILE_OP_LIBRARY << " - string (default:" << apply_orbit_file_lib_ << ")"
                << std::endl
                << PARAMETER_ID_BACKGEOCODING_OP_LIBRARY << " - string (default:" << backgeocoding_lib_ << ")"
                << std::endl
                << PARAMETER_ID_COHERENCE_OP_LIBRARY << " - string (default:" << coherence_lib_ << ")" << std::endl
                << PARAMETER_ID_DEBURST_OP_LIBRARY << " - string (default:" << deburst_lib_ << ")" << std::endl
                << PARAMETER_ID_RANGE_DOPPLER_TERRAIN_CORRECTION_LIBRARY
                << " - string (default:" << range_doppler_terrain_correction_lib_ << ")" << std::endl;

    const std::vector<std::string> routine_algs{coherence_lib_, range_doppler_terrain_correction_lib_};

    for (const auto& alg_lib_name : routine_algs) {
        const AlgorithmLoadGuard alg_guard{alg_lib_name};
        help_stream << alg_guard.GetInstanceHandle()->GetArgumentsHelp() << std::endl;
    }

    return help_stream.str();
}

void CoherenceEstimationRoutineExecute::PrintProcessingParameters() const {
    std::cout << EXECUTOR_NAME << " processing parameters:" << std::endl
              << PARAMETER_ID_APPLY_ORBIT_FILE_OP_LIBRARY << " " << apply_orbit_file_lib_ << std::endl
              << PARAMETER_ID_BACKGEOCODING_OP_LIBRARY << " " << backgeocoding_lib_ << std::endl
              << PARAMETER_ID_COHERENCE_OP_LIBRARY << " " << coherence_lib_ << std::endl
              << PARAMETER_ID_DEBURST_OP_LIBRARY << " " << deburst_lib_ << std::endl
              << PARAMETER_ID_RANGE_DOPPLER_TERRAIN_CORRECTION_LIBRARY << " " << range_doppler_terrain_correction_lib_
              << std::endl;
}

}  // namespace alus