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

#include "terrain_correction_executor.h"

#include <exception>
#include <memory>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "alg_bond.h"
#include "alus_log.h"
#include "srtm3_elevation_model.h"
#include "terrain_correction.h"
#include "terrain_correction_metadata.h"

namespace {
constexpr std::string_view PARAMETER_ID_AVG_SCENE_HEIGHT{"use_avg_scene_height"};
constexpr bool AVG_SCENE_HEIGHT_NOT_USED_VALUE{false};
}  // namespace

namespace alus::terraincorrection {
TerrainCorrectionExecutor::TerrainCorrectionExecutor() : use_avg_scene_height_{AVG_SCENE_HEIGHT_NOT_USED_VALUE} {}

int TerrainCorrectionExecutor::Execute() {
    try {
        const int default_band_id{1};
        PrintProcessingParameters();

        const auto metadata_dim_file = metadata_dim_files_.at(0);
        metadata_dimap_data_path_ = metadata_dim_file.substr(0, metadata_dim_file.length() - 4) + ".data";
        Metadata metadata(metadata_dim_file, metadata_dimap_data_path_ + "/tie_point_grids/latitude.img",
                          metadata_dimap_data_path_ + "/tie_point_grids/longitude.img");

        if (srtm3_manager_ == nullptr && !use_avg_scene_height_) {
            LOGE << "SRTM3 manager is not supplied and average scene height is not enabled, for running Range Doppler "
                    "Terrain Correction with DEM it must "
                    "be supplied";
            return 1;
        }
        if (srtm3_manager_ == nullptr) {
            std::unique_ptr<TerrainCorrection> tc{nullptr};
            if (input_datasets_.size() == 1 && input_datasets_.at(0) != nullptr) {
                Dataset<double> input(this->input_datasets_.at(0));
                tc = std::make_unique<TerrainCorrection>(std::move(input), metadata.GetMetadata(),
                                                         metadata.GetLatTiePointGrid(), metadata.GetLonTiePointGrid(),
                                                         nullptr, 0, default_band_id, use_avg_scene_height_);
            } else {
                Dataset<double> input(this->input_dataset_names_.at(0));
                tc = std::make_unique<TerrainCorrection>(std::move(input), metadata.GetMetadata(),
                                                         metadata.GetLatTiePointGrid(), metadata.GetLonTiePointGrid(),
                                                         nullptr, 0, default_band_id, use_avg_scene_height_);
            }

            tc->ExecuteTerrainCorrection(output_file_name_, tile_width_, tile_height_);
        } else {
            srtm3_manager_->HostToDevice();
            auto* const srtm3_buffers = srtm3_manager_->GetSrtmBuffersInfo();
            const auto srtm3_buffers_length = srtm3_manager_->GetDeviceSrtm3TilesCount();

            std::unique_ptr<TerrainCorrection> tc{nullptr};
            if (input_datasets_.size() == 1 && input_datasets_.at(0) != nullptr) {
                Dataset<double> input(this->input_datasets_.at(0));
                tc = std::make_unique<TerrainCorrection>(std::move(input), metadata.GetMetadata(),
                                                         metadata.GetLatTiePointGrid(), metadata.GetLonTiePointGrid(),
                                                         srtm3_buffers, srtm3_buffers_length, default_band_id,
                                                         use_avg_scene_height_);
            } else {
                Dataset<double> input(this->input_dataset_names_.at(0));
                tc = std::make_unique<TerrainCorrection>(std::move(input), metadata.GetMetadata(),
                                                         metadata.GetLatTiePointGrid(), metadata.GetLonTiePointGrid(),
                                                         srtm3_buffers, srtm3_buffers_length, default_band_id,
                                                         use_avg_scene_height_);
            }

            tc->ExecuteTerrainCorrection(output_file_name_, tile_width_, tile_height_);
            srtm3_manager_->DeviceFree();
        }

    } catch (const std::exception& e) {
        LOGE << "Exception caught while running Range Doppler Terrain Correction - " << e.what();
        return 1;
    } catch (...) {
        LOGE << "Unknown exception caught while running Range Doppler Terrain Correction";
        return 2;
    }

    return 0;
}

void TerrainCorrectionExecutor::SetInputFilenames(const std::vector<std::string>& input_datasets,
                                                  const std::vector<std::string>& metadata_paths) {
    input_dataset_names_ = input_datasets;
    metadata_dim_files_ = metadata_paths;
}

void TerrainCorrectionExecutor::SetInputDataset(const std::vector<GDALDataset*>& inputs,
                                                const std::vector<std::string>& metadata_paths) {
    input_datasets_ = inputs;
    metadata_dim_files_ = metadata_paths;
}

void TerrainCorrectionExecutor::SetParameters(const app::AlgorithmParameters::Table& param_values) {
    if (const auto use_avg_scene_height = param_values.find(PARAMETER_ID_AVG_SCENE_HEIGHT.data());
        use_avg_scene_height != param_values.end()) {
        use_avg_scene_height_ = std::stoi(use_avg_scene_height->second) != 0;
    }
    (void)param_values;
}

void TerrainCorrectionExecutor::SetSrtm3Manager(snapengine::Srtm3ElevationModel* manager) { srtm3_manager_ = manager; }

void TerrainCorrectionExecutor::SetTileSize(size_t width, size_t height) {
    tile_width_ = width;
    tile_height_ = height;
}

void TerrainCorrectionExecutor::SetOutputFilename(const std::string& output_name) { output_file_name_ = output_name; }

std::string TerrainCorrectionExecutor::GetArgumentsHelp() const {
    std::stringstream help_stream;
    help_stream << "Range Doppler Terrain Correction configurable parameters: " << PARAMETER_ID_AVG_SCENE_HEIGHT
                << " - use average scene height instead of SRTM3 DEM values (default: not used)";
    return help_stream.str();
}

void TerrainCorrectionExecutor::PrintProcessingParameters() const {
    std::string processing_params_message{"Range Doppler Terrain Correction parameters: "};
    processing_params_message += PARAMETER_ID_AVG_SCENE_HEIGHT;
    if (use_avg_scene_height_ == AVG_SCENE_HEIGHT_NOT_USED_VALUE) {
        processing_params_message += " not used";
    } else {
        processing_params_message += " used";
    }
    LOGI << processing_params_message;
}

}  // namespace alus::terraincorrection