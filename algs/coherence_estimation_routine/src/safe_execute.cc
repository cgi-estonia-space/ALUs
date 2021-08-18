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
#include "coh_tiles_generator.h"
#include "coherence_calc_cuda.h"
#include "coregistration_controller.h"
#include "cuda_algorithm_runner.h"
#include "custom/gdal_image_reader.h"
#include "custom/gdal_image_writer.h"
#include "gdal_tile_reader.h"
#include "gdal_tile_writer.h"
#include "s1tbx-commons/s_a_r_geocoding.h"
#include "snap-core/datamodel/band.h"
#include "snap-core/datamodel/product_data.h"
#include "snap-engine-utilities/datamodel/metadata/abstract_metadata.h"
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
        GDALDataset* coreg_dataset = nullptr;
        {
            const auto coreg_start = std::chrono::steady_clock::now();
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
            coreg_dataset = coreg.GetOutputDataset();
            coreg.ReleaseOutputDataset();
            LOGI << "S-1 TOPS Coregistration done - "
                 << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() -
                                                                          coreg_start)
                        .count()
                 << "ms";
            if (write_intermediate_files_) {
                LOGI << "Coregstration output @ " << cor_output_file;
                GeoTiffWriteFile(coreg_dataset, cor_output_file);
            }
        }
        // do not clear srtm from gpu as TC use it, and the new coherence takes less gpu memory
        // srtm3_manager_->DeviceFree();

        std::string coh_output_file = boost::filesystem::change_extension(cor_output_file, "").string() + "_coh.tif";
        GDALDataset* coh_dataset = nullptr;
        {
            const auto coh_start = std::chrono::steady_clock::now();
            const auto near_range_on_left = s1tbx::SARGeocoding::IsNearRangeOnLeft(
                main_product->GetTiePointGrid("incident_angle"), main_product->GetSceneRasterWidth());

            alus::coherence_cuda::MetaData meta_master{
                near_range_on_left, snapengine::AbstractMetadata::GetAbstractedMetadata(main_product), orbit_degree_};
            alus::coherence_cuda::MetaData meta_slave{
                near_range_on_left, snapengine::AbstractMetadata::GetAbstractedMetadata(secondary_product),
                orbit_degree_};

            std::vector<int> band_map{1, 2, 3, 4};
            std::vector<int> band_map_out{1};
            int band_count_in = 4;
            int band_count_out = 1;

            if (coherence_window_azimuth_ == 0) {
                // derived from pixel spacings
                coherence_window_azimuth_ =
                    static_cast<int>(std::round(coherence_window_range_ * meta_master.GetRangeAzimuthSpacingRatio()));
            }
            LOGI << "coherence window:(" << coherence_window_range_ << ", " << coherence_window_azimuth_ << ")";

            if (subtract_flat_earth_phase_) {
                LOGI << "substract flat earth phase: " << srp_polynomial_degree_ << ", " << srp_number_points_ << ", "
                     << orbit_degree_;
            }

            alus::coherence_cuda::GdalTileReader coh_data_reader{coreg_dataset, band_map, band_count_in, true};

            alus::BandParams band_params{band_map_out,
                                         band_count_out,
                                         coh_data_reader.GetBandXSize(),
                                         coh_data_reader.GetBandYSize(),
                                         coh_data_reader.GetBandXMin(),
                                         coh_data_reader.GetBandYMin()};

            alus::coherence_cuda::GdalTileWriter coh_data_writer{GetGDALDriverManager()->GetDriverByName("MEM"),
                                                                 band_params, coh_data_reader.GetGeoTransform(),
                                                                 coh_data_reader.GetDataProjection()};

            alus::coherence_cuda::CohTilesGenerator tiles_generator{
                coh_data_reader.GetBandXSize(), coh_data_reader.GetBandYSize(), static_cast<int>(tile_width_),
                static_cast<int>(tile_height_), coherence_window_range_,        coherence_window_azimuth_};

            alus::coherence_cuda::CohWindow coh_window{coherence_window_range_, coherence_window_azimuth_};
            alus::coherence_cuda::CohCuda coherence{
                srp_number_points_, srp_polynomial_degree_, subtract_flat_earth_phase_,
                coh_window,         orbit_degree_,          meta_master,
                meta_slave};

            alus::coherence_cuda::CUDAAlgorithmRunner cuda_algo_runner{&coh_data_reader, &coh_data_writer,
                                                                       &tiles_generator, &coherence};
            cuda_algo_runner.Run();
            coh_dataset = coh_data_writer.GetGdalDataset();
            LOGI << "Coherence done - "
                 << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - coh_start)
                        .count()
                 << "ms";

            if (write_intermediate_files_) {
                LOGI << "Coherence output @ " << coh_output_file;
                GeoTiffWriteFile(coh_data_writer.GetGdalDataset(), coh_output_file);
            }
        }

        // deburst
        const auto deb_start = std::chrono::steady_clock::now();
        auto data_reader = std::make_shared<alus::snapengine::custom::GdalImageReader>();
        data_reader->TakeExternalDataset(coh_dataset);
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
                          data_reader->GetDataProjection(), true);
        debursted_product->SetImageWriter(data_writer);
        deburst_op->Compute();
        LOGI << "TOPSAR Deburst done - "
             << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - deb_start)
                    .count()
             << "ms";

        if (write_intermediate_files_) {
            LOGI << "Deburst output @ " << deb_output_file;
            GeoTiffWriteFile(data_writer->GetDataset(), deb_output_file);
        }

        // TC
        const auto tc_start = std::chrono::steady_clock::now();
        terraincorrection::Metadata metadata(debursted_product);

        // uncomment next line if srtm is unloaded from gpu after coregstration
        // srtm3_manager_->HostToDevice();

        const auto* d_srtm_3_tiles = srtm3_manager_->GetSrtmBuffersInfo();
        const size_t srtm_3_tiles_length = srtm3_manager_->GetDeviceSrtm3TilesCount();
        const int selected_band{1};
        terraincorrection::TerrainCorrection tc(data_writer->GetDataset(), metadata.GetMetadata(),
                                                metadata.GetLatTiePointGrid(), metadata.GetLonTiePointGrid(),
                                                d_srtm_3_tiles, srtm_3_tiles_length, selected_band);
        std::string tc_output_file = predefined_end_result_name.empty()
                                         ? boost::filesystem::change_extension(deb_output_file, "").string() + "_tc.tif"
                                         : predefined_end_result_name;
        tc.ExecuteTerrainCorrection(tc_output_file, tile_width_, tile_height_);
        srtm3_manager_->DeviceFree();
        LOGI << "Terrain correction done - "
             << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - tc_start)
                    .count()
             << "ms";
        LOGI << "Algorithm completed, output file @ " << tc_output_file;

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