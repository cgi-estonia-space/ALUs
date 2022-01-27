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

#include <cstdlib>
#include <sstream>
#include <string_view>

#include <boost/algorithm/string/predicate.hpp>
#include <boost/filesystem.hpp>
#include <boost/lexical_cast.hpp>

#include "alus_log.h"
#include "gdal_management.h"
#include "zip_util.h"
#include "general_utils.h"

namespace {
constexpr std::string_view EXECUTOR_NAME{"Coherence estimation routine"};
constexpr std::string_view PARAMETER_ID_MAIN_SCENE{"main_scene_identifier"};
constexpr std::string_view PARAMETER_ID_MAIN_SCENE_ORBIT_FILE{"main_scene_orbit_file"};
constexpr std::string_view PARAMETER_ID_SECONDARY_SCENE_ORBIT_FILE{"secondary_scene_orbit_file"};
constexpr std::string_view PARAMETER_ID_ORBIT_FILE_DIR{"orbit_file_dir"};
constexpr std::string_view PARAMETER_ID_SUBSWATH{"subswath"};
constexpr std::string_view PARAMETER_ID_POLARIZATION{"polarization"};
constexpr std::string_view PARAMETER_ID_SRP_NUMBER_POINTS{"srp_number_points"};
constexpr std::string_view PARAMETER_ID_SRP_POLYNOMIAL_DEGREE{"srp_polynomial_degree"};
constexpr std::string_view PARAMETER_ID_SUBTRACT_FLAT_EARTH_PHASE{"subtract_flat_earth_phase"};
constexpr std::string_view PARAMETER_ID_RG_WINDOW{"rg_window"};
constexpr std::string_view PARAMETER_ID_AZ_WINDOW{"az_window"};
constexpr std::string_view PARAMETER_ID_ORBIT_DEGREE{"orbit_degree"};
constexpr std::string_view PARAMETER_WRITE_INTERMEDIATE_FILES("wif");
constexpr std::string_view PARAMETER_ID_MAIN_SCENE_FIRST_BURST_INDEX{"main_scene_first_burst_index"};
constexpr std::string_view PARAMETER_ID_MAIN_SCENE_LAST_BURST_INDEX{"main_scene_last_burst_index"};
constexpr std::string_view PARAMETER_ID_SECONDARY_SCENE_FIRST_BURST_INDEX{"secondary_scene_first_burst_index"};
constexpr std::string_view PARAMETER_ID_SECONDARY_SCENE_LAST_BURST_INDEX{"secondary_scene_last_burst_index"};
constexpr std::string_view PARAMETER_ID_WKT_AOI{"aoi"};
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

    // 2 x SLC input = ~2.5 GB + TC output = ~1GB, mem driver do not use the cache
    // should never reach 8 GB, but left larger as optional intermediate file writing is RasterIO based for now
    alus::gdalmanagement::SetCacheMax(8e9);

    int exit_code{};
    if (utils::general::IsZipOrSafeInput(input_datasets_)) {
        exit_code = ExecuteSafe();
    } else {
        LOGE << "Did not detect a SAFE input";
        exit_code = 1;
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
        {PARAMETER_ID_WKT_AOI, wkt_aoi_}};

    for (const auto& [param, member_var] : string_parameters_var_table) {
        auto value_it = param_values.find(std::string(param));
        if (value_it != param_values.end()) {
            member_var = value_it->second;
        }
    }

    const std::map<std::string_view, int&> int_parameters_var_table{
        {PARAMETER_ID_MAIN_SCENE_FIRST_BURST_INDEX, main_scene_first_burst_index_},
        {PARAMETER_ID_MAIN_SCENE_LAST_BURST_INDEX, main_scene_last_burst_index_},
        {PARAMETER_ID_SECONDARY_SCENE_FIRST_BURST_INDEX, secondary_scene_first_burst_index_},
        {PARAMETER_ID_SECONDARY_SCENE_LAST_BURST_INDEX, secondary_scene_last_burst_index_}};

    for (const auto& [param, member_var] : int_parameters_var_table) {
        auto value_it = param_values.find(std::string(param));
        if (value_it != param_values.end()) {
            member_var = boost::lexical_cast<int>(value_it->second);
        }
    }

    alg_params_ = param_values;

    ParseCoherenceParams();
    ParseOutputParams();
}

void CoherenceEstimationRoutineExecute::SetSrtm3Manager(snapengine::Srtm3ElevationModel* manager) {
    srtm3_manager_ = manager;
}

void CoherenceEstimationRoutineExecute::SetEgm96Manager(snapengine::EarthGravitationalModel96* manager) {
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
    help_stream << EXECUTOR_NAME << " parameters:\n"
                << PARAMETER_ID_MAIN_SCENE << " - string Full or partial identifier of main scene of input datasets\n"
                << PARAMETER_ID_MAIN_SCENE_ORBIT_FILE << " - string Full path of the main scene's orbit file\n"
                << PARAMETER_ID_SECONDARY_SCENE_ORBIT_FILE
                << " - string Full path of the secondary scene's orbit file\n"
                << PARAMETER_ID_ORBIT_FILE_DIR
                << " - string ESA SNAP compatible root folder of orbit files. "
                   "For example: /home/user/.snap/auxData/Orbits/Sentinel-1/POEORB/\n"
                << PARAMETER_ID_SUBSWATH << " - string Subswath to process - valid values: IW1, IW2, IW3\n"
                << PARAMETER_ID_POLARIZATION << " - string Polarization to process - valid value: VV, VH\n"
                << PARAMETER_ID_MAIN_SCENE_FIRST_BURST_INDEX
                << " - main scene first burst index, omit to process full subswath" << std::endl
                << PARAMETER_ID_MAIN_SCENE_LAST_BURST_INDEX
                << " - main scene last burst index, omit to process full subswath" << std::endl
                << PARAMETER_ID_SECONDARY_SCENE_FIRST_BURST_INDEX
                << " - secondary scene first burst index, omit to process full subswath" << std::endl
                << PARAMETER_ID_SECONDARY_SCENE_LAST_BURST_INDEX
                << " - secondary scene last burst index, omit to process full subswath" << std::endl
                << PARAMETER_ID_WKT_AOI << " - Area Of Interest WKT polygon, overrules first and last burst indexes"
                << std::endl;

    help_stream << GetCoherenceHelp();
    help_stream
        << "use_avg_scene_height - average scene height to be used instead of DEM values during terrain correction"
        << std::endl;
    help_stream << PARAMETER_WRITE_INTERMEDIATE_FILES << " - write intermediate files - true/false (default:false)"
                << std::endl;

    return help_stream.str();
}

std::string CoherenceEstimationRoutineExecute::GetCoherenceHelp() const {
    std::stringstream help_stream;
    // clang-format off
    help_stream << "Coherence configuration options:\n"
                << PARAMETER_ID_SRP_NUMBER_POINTS << " - unsigned integer (default:" << srp_number_points_ << ")\n"

                << PARAMETER_ID_SRP_POLYNOMIAL_DEGREE << " - unsigned integer (default:" << srp_polynomial_degree_
                << ")\n"
                << PARAMETER_ID_SUBTRACT_FLAT_EARTH_PHASE
                << " - true/false (default:" << (subtract_flat_earth_phase_ ? "true" : "false") << ")\n"
                << PARAMETER_ID_RG_WINDOW << " - range window size in pixels (default:" << coherence_window_range_
                << ")\n"
                << PARAMETER_ID_AZ_WINDOW
                << " - azimuth window size in pixels, if zero derived from range window (default:" <<
                    coherence_window_azimuth_  << ")\n"
                << PARAMETER_ID_ORBIT_DEGREE << " - unsigned integer (default:" << orbit_degree_ << ")\n";
    // clang-format on

    return help_stream.str();
}

void CoherenceEstimationRoutineExecute::ParseCoherenceParams() {
    auto orb_deg_it = alg_params_.find(PARAMETER_ID_ORBIT_DEGREE.data());
    if (orb_deg_it != alg_params_.end()) {
        orbit_degree_ = std::stoi(orb_deg_it->second);
    }
    auto az_window_it = alg_params_.find(PARAMETER_ID_AZ_WINDOW.data());
    if (az_window_it != alg_params_.end()) {
        coherence_window_azimuth_ = std::stoi(az_window_it->second);
    }
    auto rg_window_it = alg_params_.find(PARAMETER_ID_RG_WINDOW.data());
    if (rg_window_it != alg_params_.end()) {
        coherence_window_range_ = std::stoi(rg_window_it->second);
    }

    auto sfep_it = alg_params_.find(PARAMETER_ID_SUBTRACT_FLAT_EARTH_PHASE.data());
    if (sfep_it != alg_params_.end()) {
        if (boost::iequals(sfep_it->second, subtract_flat_earth_phase_ ? "false" : "true")) {
            subtract_flat_earth_phase_ = !subtract_flat_earth_phase_;
        }
    }
    auto srp_poly_it = alg_params_.find(PARAMETER_ID_SRP_POLYNOMIAL_DEGREE.data());
    if (srp_poly_it != alg_params_.end()) {
        srp_polynomial_degree_ = std::stoi(srp_poly_it->second);
    }

    auto srp_nr_it = alg_params_.find(PARAMETER_ID_SRP_NUMBER_POINTS.data());
    if (srp_nr_it != alg_params_.end()) {
        srp_number_points_ = std::stoi(srp_nr_it->second);
    }
}

void CoherenceEstimationRoutineExecute::PrintProcessingParameters() const {
    LOGI << EXECUTOR_NAME << " processing parameters:" << std::endl
         << "Main scene - " << main_scene_file_path_ << std::endl
         << "Secondary scene - " << secondary_scene_file_path_ << std::endl
         << PARAMETER_ID_MAIN_SCENE_ORBIT_FILE << " - " << main_scene_orbit_file_ << std::endl
         << PARAMETER_ID_SECONDARY_SCENE_ORBIT_FILE << " - " << secondary_scene_orbit_file_ << std::endl
         << PARAMETER_ID_ORBIT_FILE_DIR << " - " << orbit_file_dir_ << std::endl
         << PARAMETER_ID_SUBSWATH << " - " << subswath_ << std::endl
         << PARAMETER_ID_POLARIZATION << " - " << polarization_ << std::endl
         << PARAMETER_WRITE_INTERMEDIATE_FILES << " - " << write_intermediate_files_ << std::endl
         << PARAMETER_ID_MAIN_SCENE_FIRST_BURST_INDEX << " - " << main_scene_first_burst_index_ << std::endl
         << PARAMETER_ID_MAIN_SCENE_LAST_BURST_INDEX << " - " << main_scene_last_burst_index_ << std::endl
         << PARAMETER_ID_SECONDARY_SCENE_FIRST_BURST_INDEX << " - " << secondary_scene_first_burst_index_ << std::endl
         << PARAMETER_ID_SECONDARY_SCENE_LAST_BURST_INDEX << " - " << secondary_scene_last_burst_index_ << std::endl
         << PARAMETER_ID_WKT_AOI << " - " << wkt_aoi_ << std::endl;
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
void CoherenceEstimationRoutineExecute::ParseOutputParams() {
    if (const auto write_intermediate_files = alg_params_.find(PARAMETER_WRITE_INTERMEDIATE_FILES.data());
        write_intermediate_files != alg_params_.end()) {
        write_intermediate_files_ = boost::iequals(write_intermediate_files->second, "true");
    }
}
}  // namespace alus

extern "C" {
alus::AlgBond* CreateAlgorithm() { return new alus::CoherenceEstimationRoutineExecute(); }  // NOSONAR

void DeleteAlgorithm(alus::AlgBond* instance) { delete (alus::CoherenceEstimationRoutineExecute*)instance; }  // NOSONAR
}
