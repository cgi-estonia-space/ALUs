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

#include <string_view>
#include <sstream>

#include <boost/filesystem.hpp>

#include "alus_log.h"
#include "backgeocoding_bond.h"
#include "coherence_execute.h"
#include "terrain_correction_executor.h"

namespace {
constexpr std::string_view EXECUTOR_NAME{"Coherence estimation routine"};
constexpr std::string_view PARAMETER_ID_MAIN_SCENE{"main_scene_identifier"};
constexpr std::string_view PARAMETER_ID_MAIN_SCENE_ORBIT_FILE{"main_scene_orbit_file"};
constexpr std::string_view PARAMETER_ID_SECONDARY_SCENE_ORBIT_FILE{"secondary_scene_orbit_file"};
constexpr std::string_view PARAMETER_ID_ORBIT_FILE_DIR{"orbit_file_dir"};
constexpr std::string_view PARAMETER_ID_SUBSWATH{"subswath"};
constexpr std::string_view PARAMETER_ID_POLARIZATION{"polarization"};
constexpr std::string_view PARAMETER_ID_COH_TC_METADATA_DIM{"coherence_terrain_correction_metadata"};
}  // namespace

namespace alus {

int CoherenceEstimationRoutineExecute::Execute() {

    if (input_datasets_.size() != 2) {
        LOGE << "Coherence estimation expects 2 scenes - main and secondary, currently supplied - "
                  << input_datasets_.size();
        return 3;
    }

    for (const auto& input_ds : input_datasets_) {
        if (auto main_scene = input_ds.find(main_scene_file_id_); main_scene != std::string::npos) {
            main_scene_file_path_ = input_ds;
        } else {
            secondary_scene_file_path_ = input_ds;
        }
    }

    PrintProcessingParameters();

    GDALSetCacheMax64(4e9); // GDAL Cache 4GB, enough for for whole swath input + output

    int exit_code{};
    if (IsSafeInput()) {
        exit_code = ExecuteSafe();
    } else {
        exit_code = ExecuteGeoTiffAndBeamDimap();
    }

    return exit_code;
}

void CoherenceEstimationRoutineExecute::SetInputFilenames(const std::vector<std::string>& input_datasets,
                                                          const std::vector<std::string>& metadata_paths) {
    input_datasets_ = input_datasets;
    metadata_paths_ = metadata_paths;
}

void CoherenceEstimationRoutineExecute::SetParameters(const app::AlgorithmParameters::Table& param_values) {
    const std::map<std::string_view, std::string&> string_parameters_var_table{
        {PARAMETER_ID_MAIN_SCENE, main_scene_file_id_},
        {PARAMETER_ID_MAIN_SCENE_ORBIT_FILE, main_scene_orbit_file_},
        {PARAMETER_ID_SECONDARY_SCENE_ORBIT_FILE, secondary_scene_orbit_file_},
        {PARAMETER_ID_ORBIT_FILE_DIR, orbit_file_dir_},
        {PARAMETER_ID_SUBSWATH, subswath_},
        {PARAMETER_ID_POLARIZATION, polarization_},
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
    help_stream << EXECUTOR_NAME << " parameters:"
                << PARAMETER_ID_MAIN_SCENE << " - string Full or partial identifier of main scene of input datasets"
               
                << PARAMETER_ID_MAIN_SCENE_ORBIT_FILE << " - string Full path of the main scene's orbit file"
               
                << PARAMETER_ID_SECONDARY_SCENE_ORBIT_FILE << " - string Full path of the secondary scene's orbit file"
               
                << PARAMETER_ID_ORBIT_FILE_DIR << " - string ESA SNAP compatible root folder or orbit files. "
                                                  "For example: /home/user/.snap/auxData/Orbits/Sentinel-1/POEORB/"
               
                << PARAMETER_ID_SUBSWATH << " - string Subswath to process - valid values: IW1, IW2, IW3"
                << PARAMETER_ID_POLARIZATION << " - string Polarization to process - valid value: VV, VH";

    help_stream << backgeocoding::BackgeocodingBond().GetArgumentsHelp();
    help_stream << CoherenceExecuter().GetArgumentsHelp();
    help_stream << terraincorrection::TerrainCorrectionExecutor().GetArgumentsHelp();

    return help_stream.str();
}

void CoherenceEstimationRoutineExecute::PrintProcessingParameters() const {
    LOGI << EXECUTOR_NAME << " processing parameters:"
              << PARAMETER_ID_MAIN_SCENE << " " << main_scene_file_id_
              << PARAMETER_ID_MAIN_SCENE_ORBIT_FILE << " " << main_scene_orbit_file_
              << PARAMETER_ID_SECONDARY_SCENE_ORBIT_FILE << " " << secondary_scene_orbit_file_
              << PARAMETER_ID_ORBIT_FILE_DIR << " " << orbit_file_dir_
              << PARAMETER_ID_SUBSWATH << " " << subswath_
              << PARAMETER_ID_POLARIZATION << " " << polarization_
              << "Main scene - " << main_scene_file_path_
              << "Secondary scene - " << secondary_scene_file_path_;
}

bool CoherenceEstimationRoutineExecute::IsSafeInput() const {

    for (const auto& ds : input_datasets_) {
        const auto ext = boost::filesystem::path(ds).extension().string();
        if (ext != ".SAFE" && ext != ".safe") {
            return false;
        }
    }
    return true;
}
}  // namespace alus

extern "C" {
alus::AlgBond* CreateAlgorithm() { return new alus::CoherenceEstimationRoutineExecute(); } //NOSONAR

void DeleteAlgorithm(alus::AlgBond* instance) { delete (alus::CoherenceEstimationRoutineExecute*)instance; } //NOSONAR
}
