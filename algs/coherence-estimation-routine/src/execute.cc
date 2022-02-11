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

#include "execute.h"

#include <filesystem>
#include <future>
#include <string>

#include "abstract_metadata.h"
#include "algorithm_exception.h"
#include "alus_log.h"
#include "coh_tiles_generator.h"
#include "coh_window.h"
#include "coherence_calc_cuda.h"
#include "coregistration_controller.h"
#include "cuda_algorithm_runner.h"
#include "dem_assistant.h"
#include "gdal_image_reader.h"
#include "gdal_image_writer.h"
#include "gdal_management.h"
#include "gdal_tile_writer.h"
#include "meta_data.h"
#include "product.h"
#include "terrain_correction.h"
#include "terrain_correction_metadata.h"
#include "topsar_deburst_op.h"

namespace {
constexpr size_t GPU_DEVICE_TIMEOUT_SECONDS{10};
constexpr size_t GDAL_CACHE_SIZE{static_cast<size_t>(4e9)};
}

namespace alus::coherenceestimationroutine {

Execute::Execute(Parameters params, const std::vector<std::string>& dem_files)
    : params_{std::move(params)}, dem_files_{dem_files} {
    alus::gdalmanagement::Initialize();
    alus::gdalmanagement::SetCacheMax(GDAL_CACHE_SIZE);
}

void Execute::Run(alus::cuda::CudaInit& cuda_init, size_t) const {
    std::string result_stem{};
    std::string predefined_end_result_name{};
    std::string output_folder{};

    std::shared_ptr<app::DemAssistant> dem_assistant{nullptr};
    auto cuda_init_dem_load = std::async(std::launch::async, [&dem_assistant, &cuda_init, this]() {
        // Eagerly load DEM files in case there will be only single GPU - this will speed up things.
        dem_assistant = app::DemAssistant::CreateFormattedSrtm3TilesOnGpuFrom(this->dem_files_);
        dem_assistant->GetSrtm3Manager()->HostToDevice();

        while (!cuda_init.IsFinished());
        cuda_init.CheckErrors();

        // Here load DEM files onto other GPU devices too if support added.
    });

    if (std::filesystem::is_directory(std::filesystem::path(params_.output))) {
        // For example "/tmp/" is given. Result would be "/tmp/MAIN_SCENE_ID_Orb_Split_Stack_Coh_TC.tif"
        output_folder = params_.output + "/";
        result_stem = std::filesystem::path(params_.input_reference).filename().stem().string();
    } else {
        output_folder = std::filesystem::path(params_.output).parent_path().string() + "/";
        predefined_end_result_name = params_.output;
        result_stem = std::filesystem::path(params_.output).stem().string();
    }

    std::string cor_output_file = output_folder + result_stem + "_Orb_Stack";
    std::shared_ptr<snapengine::Product> main_product{};
    std::shared_ptr<snapengine::Product> secondary_product{};
    std::vector<GDALDataset*> coreg_output_datasets;
    {
        const auto coreg_start = std::chrono::steady_clock::now();

        coregistration::Coregistration coreg{params_.orbit_dir};
        coregistration::Coregistration::Parameters coreg_params{};
        coreg_params.main_scene_file = params_.input_reference;
        coreg_params.main_orbit_file = params_.orbit_reference;
        coreg_params.main_scene_first_burst_index = params_.burst_index_start_reference;
        coreg_params.main_scene_last_burst_index = params_.burst_index_last_reference;
        coreg_params.secondary_scene_file = params_.input_secondary;
        coreg_params.secondary_orbit_file = params_.orbit_secondary;
        coreg_params.secondary_scene_first_burst_index = params_.burst_index_start_secondary;
        coreg_params.secondary_scene_last_burst_index = params_.burst_index_last_secondary;
        coreg_params.polarisation = params_.polarisation;
        coreg_params.subswath = params_.subswath;
        coreg_params.aoi = params_.aoi;
        coreg_params.output_file = cor_output_file;

        coreg.Initialize(coreg_params);
        const auto coreg_middle = std::chrono::steady_clock::now();
        if (cuda_init_dem_load.wait_for(std::chrono::seconds(GPU_DEVICE_TIMEOUT_SECONDS)) ==
            std::future_status::timeout) {
            THROW_ALGORITHM_EXCEPTION(ALG_NAME, "CUDA device init and DEM loading has exceeded timeout");
        }
        cuda_init_dem_load.get();
        const auto cuda_device = cuda_init.GetDevices().front();
        cuda_device.Set();
        LOGI << "Using '" << cuda_device.GetName() << "' device nr " << cuda_device.GetDeviceNr()
             << " for calculations";

        const auto coreg_gpu_start = std::chrono::steady_clock::now();
        coreg.DoWork(dem_assistant->GetEgm96Manager()->GetDeviceValues(),
                     {dem_assistant->GetSrtm3Manager()->GetSrtmBuffersInfo(),
                      dem_assistant->GetSrtm3Manager()->GetDeviceSrtm3TilesCount()});
        main_product = coreg.GetMasterProduct();
        secondary_product = coreg.GetSlaveProduct();
        auto coreg_target_dataset = coreg.GetTargetDataset();
        coreg_output_datasets = coreg_target_dataset->GetDataset();
        coreg_target_dataset->ReleaseDataset();
        LOGI << "S-1 TOPS Coregistration done - "
             << std::chrono::duration_cast<std::chrono::milliseconds>(
                    (std::chrono::steady_clock::now() - coreg_gpu_start) + (coreg_middle - coreg_start))
                    .count()
             << "ms";
        if (params_.wif) {
            LOGI << "Coregstration output base @ " << cor_output_file;
            GeoTiffWriteFile(coreg_output_datasets.at(0), cor_output_file + "_mst_I");
            GeoTiffWriteFile(coreg_output_datasets.at(1), cor_output_file + "_mst_Q");
            GeoTiffWriteFile(coreg_output_datasets.at(2), cor_output_file + "_slave_I");
            GeoTiffWriteFile(coreg_output_datasets.at(3), cor_output_file + "_slave_Q");
        }
    }

    std::string coh_output_file = boost::filesystem::change_extension(cor_output_file, "").string() + "_coh.tif";
    GDALDataset* coh_dataset = nullptr;
    {
        const auto coh_start = std::chrono::steady_clock::now();
        s1tbx::Sentinel1Utils su(main_product);
        const bool near_range_on_left = su.GetSubSwath().at(0)->IsNearRangeOnLeft();
        const double avg_incidence_angle = su.GetSubSwath().at(0)->CalcAvgIncidenceAngle();

        alus::coherence_cuda::MetaData meta_master{near_range_on_left,
                                                   snapengine::AbstractMetadata::GetAbstractedMetadata(main_product),
                                                   static_cast<int>(params_.orbit_degree), avg_incidence_angle};
        alus::coherence_cuda::MetaData meta_slave{
            near_range_on_left, snapengine::AbstractMetadata::GetAbstractedMetadata(secondary_product),
            static_cast<int>(params_.orbit_degree), avg_incidence_angle};

        std::vector<int> band_map_out{1};
        int band_count_out = 1;

        auto coh_az_win = static_cast<int>(params_.az_window);
        if (params_.az_window == 0) {
            // derived from pixel spacings
            coh_az_win = static_cast<int>(
                std::round(static_cast<double>(params_.rg_window) * meta_master.GetRangeAzimuthSpacingRatio()));
        }
        LOGI << "coherence window:(" << params_.rg_window << ", " << coh_az_win << ")";

        if (params_.subtract_flat_earth) {
            LOGI << "substract flat earth phase: " << params_.srp_polynomial_degree << ", " << params_.srp_number_points
                 << ", " << params_.orbit_degree;
        }

        alus::coherence_cuda::GdalTileReader coh_data_reader{coreg_output_datasets};

        alus::BandParams band_params{band_map_out,
                                     band_count_out,
                                     coh_data_reader.GetBandXSize(),
                                     coh_data_reader.GetBandYSize(),
                                     coh_data_reader.GetBandXMin(),
                                     coh_data_reader.GetBandYMin()};

        alus::coherence_cuda::GdalTileWriter coh_data_writer{
            GetGDALDriverManager()->GetDriverByName("MEM"), band_params, {}, {}};

        const auto total_dimension_edge = 4096;
        const auto x_range_tile_size = static_cast<int>(
            (static_cast<double>(params_.rg_window) / static_cast<double>(params_.rg_window + coh_az_win)) *
            total_dimension_edge);
        const auto y_az_tile_size = total_dimension_edge - x_range_tile_size;
        alus::coherence_cuda::CohTilesGenerator tiles_generator{coh_data_reader.GetBandXSize(),
                                                                coh_data_reader.GetBandYSize(),
                                                                x_range_tile_size,
                                                                y_az_tile_size,
                                                                static_cast<int>(params_.rg_window),
                                                                coh_az_win};

        alus::coherence_cuda::CohWindow coh_window{static_cast<int>(params_.rg_window), coh_az_win};
        alus::coherence_cuda::CohCuda coherence{static_cast<int>(params_.srp_number_points),
                                                static_cast<int>(params_.srp_polynomial_degree),
                                                params_.subtract_flat_earth,
                                                coh_window,
                                                static_cast<int>(params_.orbit_degree),
                                                meta_master,
                                                meta_slave};

        alus::coherence_cuda::CUDAAlgorithmRunner cuda_algo_runner{&coh_data_reader, &coh_data_writer, &tiles_generator,
                                                                   &coherence};
        cuda_algo_runner.Run();
        coh_dataset = coh_data_writer.GetGdalDataset();
        LOGI << "Coherence done - "
             << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - coh_start)
                    .count()
             << "ms";

        if (params_.wif) {
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
        for (const auto& band : bands) {
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
         << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - deb_start).count()
         << "ms";

    if (params_.wif) {
        LOGI << "Deburst output @ " << deb_output_file;
        GeoTiffWriteFile(data_writer->GetDataset(), deb_output_file);
    }

    // TC
    const auto tc_start = std::chrono::steady_clock::now();
    terraincorrection::Metadata metadata(debursted_product);

    const auto* d_srtm_3_tiles = dem_assistant->GetSrtm3Manager()->GetSrtmBuffersInfo();
    const size_t srtm_3_tiles_length = dem_assistant->GetSrtm3Manager()->GetDeviceSrtm3TilesCount();
    const int selected_band{1};
    auto* tc_in_dataset = data_writer->GetDataset();
    const auto total_dimension_edge = 4096;
    const auto x_tile_size =
        static_cast<int>((tc_in_dataset->GetRasterXSize() /
                          static_cast<double>(tc_in_dataset->GetRasterXSize() + tc_in_dataset->GetRasterYSize())) *
                         total_dimension_edge);
    const auto y_tile_size = total_dimension_edge - x_tile_size;
    terraincorrection::TerrainCorrection tc(tc_in_dataset, metadata.GetMetadata(), metadata.GetLatTiePointGrid(),
                                            metadata.GetLonTiePointGrid(), d_srtm_3_tiles, srtm_3_tiles_length,
                                            selected_band);
    std::string tc_output_file = predefined_end_result_name.empty()
                                     ? boost::filesystem::change_extension(deb_output_file, "").string() + "_tc.tif"
                                     : predefined_end_result_name;
    tc.ExecuteTerrainCorrection(tc_output_file, x_tile_size, y_tile_size);
    dem_assistant->GetSrtm3Manager()->DeviceFree();
    LOGI << "Terrain correction done - "
         << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - tc_start).count()
         << "ms";
    LOGI << "Algorithm completed, output file @ " << tc_output_file;
}

Execute::~Execute() { alus::gdalmanagement::Deinitialize(); }

}  // namespace alus::coherenceestimationroutine
