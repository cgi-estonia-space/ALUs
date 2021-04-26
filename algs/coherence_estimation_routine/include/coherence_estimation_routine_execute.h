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

    // TODO: These default values could be assigned via CMake generated header file by using the operations' shared
    // library target name.
    std::string apply_orbit_file_lib_{"apply-orbit-file"};
    std::string backgeocoding_lib_{"backgeocoding"};
    std::string coherence_lib_{"coherence"};
    std::string deburst_lib_{"deburst"};
    std::string range_doppler_terrain_correction_lib_{"terrain-correction"};

    std::vector<std::string> input_datasets_{};
    std::vector<std::string> metadata_paths_{};
    snapengine::Srtm3ElevationModel* srtm3_manager_{};
    const snapengine::EarthGravitationalModel96* egm96_manager_{};
    size_t tile_width_{};
    size_t tile_height_{};
    std::string output_name_{};
    app::AlgorithmParameters::Table alg_params_;
    std::string coherence_terrain_correction_metadata_param_{};
};
}

extern "C" {
alus::AlgBond* CreateAlgorithm() { return new alus::CoherenceEstimationRoutineExecute(); } //NOSONAR

void DeleteAlgorithm(alus::AlgBond* instance) { delete (alus::CoherenceEstimationRoutineExecute*)instance; } //NOSONAR
}
