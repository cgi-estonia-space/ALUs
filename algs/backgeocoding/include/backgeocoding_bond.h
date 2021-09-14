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
#include <vector>

#include "alg_bond.h"
#include "algorithm_parameters.h"
#include "earth_gravitational_model96.h"
#include "srtm3_elevation_model.h"

namespace alus::backgeocoding {

class BackgeocodingBond : public AlgBond {
public:
    BackgeocodingBond() = default;

    void SetInputFilenames([[maybe_unused]] const std::vector<std::string>& input_datasets,
                           [[maybe_unused]] const std::vector<std::string>& metadata_paths) override;
    void SetParameters(const app::AlgorithmParameters::Table& param_values) override;
    void SetSrtm3Manager(snapengine::Srtm3ElevationModel* manager) override;
    void SetEgm96Manager(snapengine::EarthGravitationalModel96* manager) override;
    void SetTileSize(size_t width, size_t height) override;
    void SetOutputFilename([[maybe_unused]] const std::string& output_name) override;
    int Execute() override;

    [[nodiscard]] std::string GetArgumentsHelp() const override;

    ~BackgeocodingBond() override = default;

private:
    std::vector<std::string> input_dataset_filenames_{};
    std::vector<std::string> input_metadata_filenames_{};
    std::string output_filename_{};

    std::string master_dataset_arg_;
    std::string master_metadata_arg_;

    bool use_elevation_mask_;
    snapengine::Srtm3ElevationModel* srtm3_manager_{};
    snapengine::EarthGravitationalModel96* egm96_manager_{};
};

}  // namespace alus::backgeocoding
