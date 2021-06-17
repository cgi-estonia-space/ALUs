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
#include <exception>

#include <boost/filesystem.hpp>

#include "alus_log.h"
#include "coherence_execute.h"
#include "coregistration_controller.h"
#include "custom/gdal_image_reader.h"
#include "custom/gdal_image_writer.h"
#include "snap-core/datamodel/band.h"
#include "snap-core/datamodel/product_data.h"
#include "terrain_correction.h"
#include "terrain_correction_metadata.h"
#include "topsar_deburst_op.h"

namespace alus {
int CoherenceEstimationRoutineExecute::ExecuteSafe() {
    try {
        std::string result_stem{};
        std::string predefined_end_result_name{};
        std::string output_folder{};
        if (boost::filesystem::is_directory(boost::filesystem::path(output_name_))) {
            // For example "/tmp/" is given. Result would be "/tmp/MAIN_SCENE_ID_Orb_Split_Stack_Coh_TC.tif"
            output_folder = output_name_ + "/";
            result_stem = boost::filesystem::path(main_scene_file_path_).leaf().stem().string();
        } else {
            output_folder = boost::filesystem::path(output_name_).parent_path().string() + "/";
            predefined_end_result_name = output_name_;
            result_stem = boost::filesystem::path(output_name_).stem().string();
        }

        std::string cor_output_file = output_folder + result_stem + "_Orb_Stack.tif";
        srtm3_manager_->HostToDevice();
        std::shared_ptr<snapengine::Product> main_product{};
        std::shared_ptr<snapengine::Product> secondary_product{};
        {
            const auto coreg_start = std::chrono::system_clock::now();
            coregistration::Coregistration coreg{orbit_file_dir_};
            if (!main_scene_orbit_file_.empty() || !secondary_scene_orbit_file_.empty()) {
                coreg.Initialize(main_scene_file_path_, secondary_scene_file_path_, cor_output_file, "IW1", "VV",
                                 main_scene_orbit_file_, secondary_scene_orbit_file_);
            } else {
                coreg.Initialize(main_scene_file_path_, secondary_scene_file_path_, cor_output_file, "IW1", "VV");
            }
            coreg.DoWork(egm96_manager_->GetDeviceValues(),
                         {srtm3_manager_->GetSrtmBuffersInfo(), srtm3_manager_->GetDeviceSrtm3TilesCount()});
            main_product = coreg.GetMasterProduct();
            secondary_product = coreg.GetSlaveProduct();
            LOGI << "S-1 TOPS Coregistration done - "
                 << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() -
                                                                          coreg_start).count() << "ms";
        }
        srtm3_manager_->DeviceFree();

        std::string coh_output_file = boost::filesystem::change_extension(cor_output_file, "").string() + "_coh.tif";
        {
            const auto coh_start = std::chrono::system_clock::now();
            auto alg = CoherenceExecuter();
            alg.SetInputProducts(main_product, secondary_product);
            alg.SetParameters(alg_params_);
            alg.SetTileSize(tile_width_, tile_height_);
            alg.SetInputFilenames({cor_output_file}, {});
            alg.SetOutputFilename(coh_output_file);
            const auto res = alg.Execute();
            if (res != 0) {
                LOGE << "Running Coherence operation resulted in non success execution - " << res << " -aborting.";
                return res;
            }
            LOGI << "Coherence done - "
                      << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() -
                                                                               coh_start).count() << "ms";
        }

        // deburst
        const auto deb_start = std::chrono::system_clock::now();
        auto data_reader = std::make_shared<alus::snapengine::custom::GdalImageReader>();
        data_reader->Open(coh_output_file, false, true);
        main_product->SetImageReader(data_reader);
        {
            const auto& bands = main_product->GetBands();
            for (auto& band : bands) {
                main_product->RemoveBand(band);
            }
        }
        main_product->AddBand(std::make_shared<snapengine::Band>("coh_0", snapengine::ProductData::TYPE_FLOAT32,
                                                                 main_product->GetSceneRasterWidth(),
                                                                 main_product->GetSceneRasterHeight()));
        auto deburst_op = alus::s1tbx::TOPSARDeburstOp::CreateTOPSARDeburstOp(main_product);
        auto debursted_product = deburst_op->GetTargetProduct();
        std::string deb_output_file = boost::filesystem::change_extension(coh_output_file, "").string() + "_deb.tif";
        auto data_writer = std::make_shared<alus::snapengine::custom::GdalImageWriter>();
        data_writer->Open(deb_output_file, deburst_op->GetTargetProduct()->GetSceneRasterWidth(),
                          deburst_op->GetTargetProduct()->GetSceneRasterHeight(), data_reader->GetGeoTransform(),
                          data_reader->GetDataProjection());
        debursted_product->SetImageWriter(data_writer);
        deburst_op->Compute();
        LOGI << "TOPSAR Deburst done - "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - deb_start)
                    .count() << "ms";

        // TC
        const auto tc_start = std::chrono::system_clock::now();
        terraincorrection::Metadata metadata(debursted_product);
        Dataset<double> tc_dataset(deb_output_file);
        srtm3_manager_->HostToDevice();
        const auto* d_srtm_3_tiles = srtm3_manager_->GetSrtmBuffersInfo();
        const size_t srtm_3_tiles_length = srtm3_manager_->GetDeviceSrtm3TilesCount();
        const int selected_band{1};
        terraincorrection::TerrainCorrection tc(std::move(tc_dataset), metadata.GetMetadata(),
                                                metadata.GetLatTiePointGrid(), metadata.GetLonTiePointGrid(),
                                                d_srtm_3_tiles, srtm_3_tiles_length, selected_band);
        std::string tc_output_file = predefined_end_result_name.empty()
                                         ? boost::filesystem::change_extension(deb_output_file, "").string() + "_tc.tif"
                                         : predefined_end_result_name;
        tc.ExecuteTerrainCorrection(tc_output_file, tile_width_, tile_height_);
        srtm3_manager_->DeviceFree();
        LOGI << "Terrain correction done - "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - tc_start)
                    .count() << "ms";

    } catch (const std::exception& e) {
        LOGE << "Operation resulted in error:" << e.what() << " - aborting.";
        return 2;
    } catch (...) {
        LOGE << "Operation resulted in error. Aborting.";
        return 2;
    }

    return 0;
}

}  // namespace alus