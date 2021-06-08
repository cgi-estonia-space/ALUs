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

#include <exception>

#include <boost/filesystem.hpp>

#include "alus_log.h"
#include "backgeocoding_bond.h"
#include "coherence_execute.h"
#include "terrain_correction_executor.h"

namespace alus {
int CoherenceEstimationRoutineExecute::ExecuteGeoTiffAndBeamDimap() {
    try {
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
            LOGE << "Expecting single dim metadata file for coherence and terrain correction operators ("
                 << coherence_terrain_correction_metadata_param_ << ") not found";
            return 3;
        }

        std::string backg_output = boost::filesystem::change_extension(output_name_, "").string() + "_Stack.tif";
        {
            auto alg = backgeocoding::BackgeocodingBond();
            alg.SetParameters(alg_params_);
            alg.SetSrtm3Manager(srtm3_manager_);
            alg.SetEgm96Manager(egm96_manager_);
            alg.SetTileSize(tile_width_, tile_height_);
            alg.SetInputFilenames(input_datasets_, backgeocoding_metadata_files);
            alg.SetOutputFilename(backg_output);
            const auto res = alg.Execute();

            if (res != 0) {
                LOGE << "Running S-1 Backgeocoding resulted in non success execution - " << res << " - aborting.";
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
            const auto res = alg.Execute();

            if (res != 0) {
                LOGE << "Running Coherence operation resulted in non success execution - " << res << " - aborting.";
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
            const auto res = alg.Execute();

            if (res != 0) {
                LOGE << "Running Terrain correciton operation resulted in non success execution - " << res
                     << " - aborting.";
                return res;
            }
        }

    } catch (const std::exception& e) {
        LOGE << "Operation resulted in error:" << e.what() << "Aborting.";
        return 2;
    }

    return 0;
}

}  // namespace alus