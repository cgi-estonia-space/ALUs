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

#include <cstddef>
#include <string>

#include "alg_bond.h"
#include "algorithm_parameters.h"
#include "earth_gravitational_model96.h"
#include "pointer_holders.h"
#include "srtm3_elevation_model.h"

namespace alus {

class CoherenceEstimationRoutineExecute final : public AlgBond {
public:
    CoherenceEstimationRoutineExecute() = default;

    void SetInputFilenames(const std::vector<std::string>& input_datasets,
                           const std::vector<std::string>& metadata_paths) override;
    void SetParameters(const app::AlgorithmParameters::Table& param_values) override;
    void SetSrtm3Manager(snapengine::Srtm3ElevationModel* manager) override;
    void SetEgm96Manager(const snapengine::EarthGravitationalModel96* manager) override;
    void SetTileSize(size_t width, size_t height) override;
    void SetOutputFilename(const std::string& output_name) override;
    int Execute() override;

    [[nodiscard]] std::string GetArgumentsHelp() const override;

    ~CoherenceEstimationRoutineExecute() override = default;

private:
    void PrintProcessingParameters() const override;

    bool IsSafeInput() const;
    int ExecuteSafe();
    void ParseCoherenceParams();
    std::string GetCoherenceHelp() const;

    std::vector<std::string> input_datasets_{};
    std::vector<std::string> metadata_paths_{};
    snapengine::Srtm3ElevationModel* srtm3_manager_{};
    const snapengine::EarthGravitationalModel96* egm96_manager_{};
    size_t tile_width_{};
    size_t tile_height_{};
    std::string output_name_{};
    app::AlgorithmParameters::Table alg_params_;
    std::string main_scene_file_id_{};
    std::string main_scene_orbit_file_{};
    std::string secondary_scene_orbit_file_{};
    std::string orbit_file_dir_{};
    std::string subswath_{};
    std::string polarization_{};
    std::string coherence_terrain_correction_metadata_param_{};
    std::string main_scene_file_path_{};
    std::string secondary_scene_file_path_{};
    int srp_number_points_{501};
    int srp_polynomial_degree_{5};
    bool subtract_flat_earth_phase_{true};
    int coherence_window_range_{15};
    int coherence_window_azimuth_{0}; // if left as zero, derived from range window
    int orbit_degree_{3};
};
}