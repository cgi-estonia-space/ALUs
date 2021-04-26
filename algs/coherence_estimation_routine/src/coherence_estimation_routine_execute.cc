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

#include <chrono>
#include <dlfcn.h>
#include <iostream>
#include <map>
#include <string_view>
#include <sstream>

#include <boost/filesystem.hpp>

#include "alg_load.h"
#include "backgeocoding_bond.h"
#include "coherence_execute.h"
#include "terrain_correction_executor.h"

namespace {
constexpr std::string_view EXECUTOR_NAME{"Coherence estimation routine"};
constexpr std::string_view PARAMETER_ID_APPLY_ORBIT_FILE_OP_LIBRARY{"apply_orbit_file_lib"};
constexpr std::string_view PARAMETER_ID_BACKGEOCODING_OP_LIBRARY{"backgeocoding_lib"};
constexpr std::string_view PARAMETER_ID_COHERENCE_OP_LIBRARY{"coherence_lib"};
constexpr std::string_view PARAMETER_ID_DEBURST_OP_LIBRARY{"deburst_lib"};
constexpr std::string_view PARAMETER_ID_RANGE_DOPPLER_TERRAIN_CORRECTION_LIBRARY{"terrain_correction_lib"};
constexpr std::string_view PARAMETER_ID_COH_TC_METADATA_DIM{"coherence_terrain_correction_metadata"};
}  // namespace

namespace alus {

int CoherenceEstimationRoutineExecute::Execute() {

//    auto const po_driver = GetGDALDriverManager()->GetDriverByName("MEM"); //NOSONAR

    std::vector<std::string> coh_tc_metadata_file{};
    std::vector<std::string> backgeocoding_metadata_files{};
    for (const auto& dim : metadata_paths_) {
        if (dim.find(coherence_terrain_correction_metadata_param_) != std::string::npos) {
           coh_tc_metadata_file.push_back(dim);
        } else {
            backgeocoding_metadata_files.push_back(dim);
        }
    }

    if (coh_tc_metadata_file.size() != 1) {
        std::cerr << "Expecting single dim metadata file for coherence and terrain correction operators ("
                  << coherence_terrain_correction_metadata_param_ << ") not found" << std::endl;
        return 3;
    }

    PrintProcessingParameters();

    try {
        std::string backg_output = boost::filesystem::change_extension(output_name_, "").string() + "_Stack.tif";
        {
            auto alg = backgeocoding::BackgeocodingBond();
            alg.SetParameters(alg_params_);
            alg.SetSrtm3Manager(srtm3_manager_);
            alg.SetEgm96Manager(egm96_manager_);
            alg.SetTileSize(tile_width_, tile_height_);
            alg.SetInputFilenames(input_datasets_, backgeocoding_metadata_files);
            alg.SetOutputFilename(backg_output);
            const auto start = std::chrono::high_resolution_clock::now();
            const auto res = alg.Execute();
            const auto stop = std::chrono::high_resolution_clock::now();
            std::cout << "Backgeocoding spent "
                      << std::chrono::duration_cast<std::chrono::duration<float>>(stop - start).count() << " seconds."
                      << std::endl;

            if (res != 0) {
                std::cout << "Running S-1 Backgeocoding resulted in non success execution - " << res << std::endl
                          << "Aborting." << std::endl;
                return res;
            }
        }

        std::string coh_output = boost::filesystem::change_extension(backg_output, "").string() + "_coh.tif";
        {
            auto alg = CoherenceExecuter();
            alg.SetParameters(alg_params_);
            alg.SetTileSize(tile_width_, tile_height_);
            std::vector<std::string> input_dataset{backg_output};
            alg.SetInputFilenames(input_dataset, coh_tc_metadata_file);
            alg.SetOutputFilename(coh_output);
            const auto start = std::chrono::high_resolution_clock::now();
            const auto res = alg.Execute();
            const auto stop = std::chrono::high_resolution_clock::now();
            std::chrono::duration<float, std::milli> duration = stop - start;
            std::cout << "Coherence spent "
                      << std::chrono::duration_cast<std::chrono::duration<float>>(stop - start).count() << " seconds."
                      << std::endl;

            if (res != 0) {
                std::cout << "Running Coherence operation resulted in non success execution - " << res << std::endl
                          << "Aborting." << std::endl;
                return res;
            }
        }

        std::string tc_output = boost::filesystem::change_extension(coh_output, "").string() + "_tc.tif";
        {
            auto alg = terraincorrection::TerrainCorrectionExecutor();
            alg.SetParameters(alg_params_);
            alg.SetTileSize(tile_width_, tile_height_);
            alg.SetSrtm3Manager(srtm3_manager_);
            alg.SetEgm96Manager(egm96_manager_);
            std::vector<std::string> input_dataset{coh_output};
            alg.SetInputFilenames(input_dataset, coh_tc_metadata_file);
            alg.SetOutputFilename(tc_output);
            const auto start = std::chrono::high_resolution_clock::now();
            const auto res = alg.Execute();
            const auto stop = std::chrono::high_resolution_clock::now();
            std::chrono::duration<float, std::milli> duration = stop - start;
            std::cout << "Terrain correction spent "
                      << std::chrono::duration_cast<std::chrono::duration<float>>(stop - start).count() << " seconds."
                      << std::endl;

            if (res != 0) {
                std::cout << "Running Terrain correction operation resulted in non success execution - " << res
                          << std::endl
                          << "Aborting." << std::endl;
                return res;
            }
        }

    } catch (const std::exception& e) {
        std::cout << "Operation resulted in error:" << e.what() << std::endl << "Aborting." << std::endl;
        return 2;
    }

    return 0;
}

void CoherenceEstimationRoutineExecute::SetInputFilenames(const std::vector<std::string>& input_datasets,
                                                          const std::vector<std::string>& metadata_paths) {
    input_datasets_ = input_datasets;
    metadata_paths_ = metadata_paths;
}

void CoherenceEstimationRoutineExecute::SetParameters(const app::AlgorithmParameters::Table& param_values) {
    const std::map<std::string_view, std::string&> string_parameters_var_table{
        {PARAMETER_ID_APPLY_ORBIT_FILE_OP_LIBRARY, apply_orbit_file_lib_},
        {PARAMETER_ID_BACKGEOCODING_OP_LIBRARY, backgeocoding_lib_},
        {PARAMETER_ID_COHERENCE_OP_LIBRARY, coherence_lib_},
        {PARAMETER_ID_DEBURST_OP_LIBRARY, deburst_lib_},
        {PARAMETER_ID_RANGE_DOPPLER_TERRAIN_CORRECTION_LIBRARY, range_doppler_terrain_correction_lib_},
        {PARAMETER_ID_COH_TC_METADATA_DIM, coherence_terrain_correction_metadata_param_}};

    for (const auto& parameter : string_parameters_var_table) {
        auto value_it = param_values.find(std::string(parameter.first));
        if (value_it != param_values.end()) {
            parameter.second = value_it->second;
        }
    }

    alg_params_ = param_values;
}

void CoherenceEstimationRoutineExecute::SetSrtm3Manager(snapengine::Srtm3ElevationModel* manager) {
    srtm3_manager_ = manager;
}

void CoherenceEstimationRoutineExecute::SetEgm96Manager(const snapengine::EarthGravitationalModel96* manager) {
    egm96_manager_ = manager;
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
                << " - string (default:" << range_doppler_terrain_correction_lib_ << ")" << std::endl
                << PARAMETER_ID_COH_TC_METADATA_DIM << " - string Coherence and terrain correction metadata dim file ID"
                << std::endl;

    const std::vector<std::string> routine_algs{backgeocoding_lib_, coherence_lib_,
                                                range_doppler_terrain_correction_lib_};

    for (const auto& alg_lib_name : routine_algs) {
        try {
            const AlgorithmLoadGuard alg_guard{alg_lib_name};
            help_stream << alg_guard.GetInstanceHandle()->GetArgumentsHelp() << std::endl;
        } catch (...) {
            std::cerr << alg_lib_name << " cannot be loaded - please check input algorithms." << std::endl;
        }
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
              << PARAMETER_ID_COH_TC_METADATA_DIM << " " << coherence_terrain_correction_metadata_param_
              << std::endl;
}

}  // namespace alus
