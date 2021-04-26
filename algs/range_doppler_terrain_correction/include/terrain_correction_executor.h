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

#include <string>
#include <vector>

#include <gdal_priv.h>

#include "alg_bond.h"

namespace alus::terraincorrection {
class TerrainCorrectionExecutor : public AlgBond {
public:
    TerrainCorrectionExecutor();

    int Execute() override;

    void SetInputFilenames(const std::vector<std::string>& input_datasets,
                           const std::vector<std::string>& metadata_paths) override;

    void SetInputDataset(const std::vector<GDALDataset*>& inputs,
                         const std::vector<std::string>& metadata_paths) override;

    void SetParameters(const app::AlgorithmParameters::Table& param_values) override;

    void SetSrtm3Manager(snapengine::Srtm3ElevationModel* manager) override;

    void SetTileSize(size_t width, size_t height) override;

    void SetOutputFilename(const std::string& output_name) override;

    [[nodiscard]] std::string GetArgumentsHelp() const override;

    ~TerrainCorrectionExecutor() override = default;

private:
    void PrintProcessingParameters() const override;

    std::vector<std::string> input_dataset_names_{};
    std::vector<GDALDataset*> input_datasets_{};
    std::string metadata_dimap_data_path_{};
    std::vector<std::string> metadata_dim_files_{};
    std::string output_file_name_{};
    size_t tile_width_{};
    size_t tile_height_{};
    uint32_t avg_scene_height_{};
    snapengine::Srtm3ElevationModel* srtm3_manager_{};
};

}